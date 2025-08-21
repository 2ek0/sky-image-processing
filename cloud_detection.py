import cv2
import numpy as np
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json
import re
from multiprocessing import Pool, cpu_count
from functools import partial

# Add the draft directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from draft.create_sky_mask import create_sky_mask, load_sky_mask
from draft.sun_position_identification import SunPositionCalculator

def enhance_contrast(img_rgb, alpha=1.3, beta=10):
    """
    Enhance contrast of the image using linear transformation
    
    Args:
        img_rgb: RGB image to enhance
        alpha: Contrast factor (1.0 = no change, >1.0 = increase contrast)
        beta: Brightness offset (-255 to 255)
    
    Returns:
        Enhanced RGB image
    """
    # Apply contrast and brightness adjustment
    enhanced = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

def calculate_nrbr(img_rgb):
    """Calculate Normalized Red Blue Ratio"""
    r = img_rgb[:,:,0].astype(float)
    b = img_rgb[:,:,2].astype(float)
    return (r - b) / (r + b + 1e-6)

def get_clear_sky_image(timestamp, clear_sky_dir):
    """
    Select the most appropriate clear sky image from the library based on sun position
    
    Args:
        timestamp: datetime object of the target image
        clear_sky_dir: directory containing clear sky images
    
    Returns:
        path to the most appropriate clear sky image
    """
    # Get all clear sky images
    clear_sky_images = glob.glob(os.path.join(clear_sky_dir, "*.jpg"))
    if not clear_sky_images:
        return None
        
    # Calculate sun position for target image
    target_calculator = SunPositionCalculator(
        image_size=(1000, 1000),  # Default size, will be resized later
        image_center=(500, 500),
        image_radius=490
    )
    target_sun_pos = target_calculator.calculate_position(timestamp)
    
    # Find the clear sky image with closest sun position
    min_diff = float('inf')
    best_image = None
    
    for cs_image in clear_sky_images:
        # Extract timestamp from filename
        filename = os.path.basename(cs_image)
        filename = os.path.splitext(filename)[0]
        try:
            cs_timestamp = datetime.strptime(filename, '%Y%m%d_%H%M%S')
            # Add 8 hours since 0 hour represents 8:00 AM
            cs_timestamp = cs_timestamp.replace(hour=cs_timestamp.hour + 8)
            
            # Calculate sun position for clear sky image
            cs_calculator = SunPositionCalculator(
                image_size=(1000, 1000),
                image_center=(500, 500),
                image_radius=490
            )
            cs_sun_pos = cs_calculator.calculate_position(cs_timestamp)
            
            # Calculate difference in sun positions
            diff = np.sum(np.abs(target_sun_pos.mask - cs_sun_pos.mask))
            
            if diff < min_diff:
                min_diff = diff
                best_image = cs_image
                
        except ValueError:
            continue
    
    return best_image

def detect_sunshine_area(img_rgb, sun_position):
    """
    Enhanced sunshine area detection with better criteria
    
    Args:
        img_rgb: RGB image to analyze
        sun_position: SunPosition object containing sun position mask
        
    Returns:
        mask: Binary mask of sunshine area
    """
    # Convert to HSV for better brightness analysis
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    # Create gradient distance from sun center
    if np.sum(sun_position.mask) > 0:
        sun_center = np.argwhere(sun_position.mask > 0).mean(axis=0).astype(int)
        y, x = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        dist_from_sun = np.sqrt((x - sun_center[1])**2 + (y - sun_center[0])**2)
        
        # Normalize distance to 0-1 range
        max_dist = np.max(dist_from_sun)
        if max_dist > 0:
            norm_dist = dist_from_sun / max_dist
        else:
            norm_dist = dist_from_sun
            
        # Create kernels for adaptive sunshine detection
        small_kernel = np.ones((3, 3), np.uint8)
        large_kernel = np.ones((15, 15), np.uint8)
        
        # Dilate sun mask to capture surrounding area - two sizes for different criteria
        sun_core = sun_position.mask.copy()
        sun_region = cv2.dilate(sun_position.mask, small_kernel, iterations=2)
        sunshine_region = cv2.dilate(sun_position.mask, large_kernel, iterations=1)
        
        # Calculate NRBR in the sunshine area for additional criteria
        nrbr = calculate_nrbr(img_rgb)
        
        # IMPROVED: Multi-layered sunshine criteria
        sunshine_mask = np.zeros_like(sun_position.mask)
        
        # Core sunshine - very bright center region
        core_criteria = (
            (v_channel > 220) &  # Very high brightness
            (s_channel < 40) &   # Very low saturation
            (sun_core > 0)       # In core sun region
        )
        
        # Medium sunshine - bright with low-medium saturation
        medium_criteria = (
            (v_channel > 200) &  # High brightness
            (s_channel < 60) &   # Low saturation
            (norm_dist < 0.15) & # Close to sun center
            (sun_region > 0)     # In dilated sun region
        )
        
        # Extended sunshine - more lenient criteria for periphery
        extended_criteria = (
            (v_channel > 180) &  # Moderately high brightness
            (s_channel < 80) &   # Low-medium saturation
            (norm_dist < 0.25) & # Relatively close to sun
            (sunshine_region > 0) & # In widely dilated sun region
            (nrbr < -0.10)       # NRBR still indicates some blueness
        )
        
        # Combine all sunshine criteria
        sunshine_mask[core_criteria | medium_criteria | extended_criteria] = 255
        
        # Apply morphological operations to smooth the mask
        sunshine_mask = cv2.morphologyEx(sunshine_mask, cv2.MORPH_CLOSE, 
                                       np.ones((5, 5), np.uint8))
        
        return sunshine_mask
    
    return np.zeros_like(sun_position.mask)

def enhance_thin_cloud_detection(img_rgb, nrbr, sky_mask):
    """
    Enhanced detection of thin clouds using multiple features
    
    Args:
        img_rgb: RGB image to analyze
        nrbr: Normalized Red Blue Ratio image
        sky_mask: Binary mask of sky region
        
    Returns:
        thin_cloud_mask: Binary mask of detected thin clouds
    """
    # Convert to different color spaces for feature extraction
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Extract relevant channels
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    l_channel = lab[:,:,0]
    
    # Calculate local contrast in L channel (texture feature)
    blur = cv2.GaussianBlur(l_channel, (5, 5), 0)
    local_contrast = cv2.absdiff(l_channel, blur)
    
    # Calculate gradient magnitude of NRBR (edge feature)
    sobelx = cv2.Sobel(nrbr, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(nrbr, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient to 0-1 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # IMPROVED: More sensitive thin cloud detection with expanded criteria
    thin_cloud_mask = np.zeros_like(sky_mask)
    
    # Primary thin cloud criteria - expanded ranges
    thin_cloud_criteria = (
        (nrbr > -0.55) & (nrbr < 0.0) &  # Wider NRBR range
        (v_channel > 120) & (v_channel < 250) &  # Lower minimum brightness threshold
        (s_channel < 130) &  # Higher saturation threshold to include more thin clouds
        ((local_contrast > 1.2) | (gradient_magnitude > 0.04)) &  # More sensitive texture/edge detection
        (sky_mask > 0)  # Within sky region
    )
    thin_cloud_mask[thin_cloud_criteria] = 255
    
    # ADDED: Additional pass to catch very thin clouds based on texture and relative NRBR
    very_thin_criteria = (
        (nrbr > -0.6) & (nrbr < -0.3) &  # Special range for very thin clouds
        (local_contrast > 2.0) &  # Strong texture indicates cloud structure
        (v_channel > 140) & (v_channel < 240) &
        (sky_mask > 0)
    )
    thin_cloud_mask[very_thin_criteria] = 255
    
    # Clean up the mask
    thin_cloud_mask = cv2.morphologyEx(thin_cloud_mask, cv2.MORPH_CLOSE, 
                                      np.ones((5, 5), np.uint8))
    
    return thin_cloud_mask

def detect_additional_thin_clouds(img_rgb, nrbr, sky_mask):
    """
    Specialized detector focused solely on thin cloud features using multiple indicators
    
    Args:
        img_rgb: RGB image to analyze
        nrbr: Normalized Red Blue Ratio image
        sky_mask: Binary mask of sky region
        
    Returns:
        thin_cloud_mask: Binary mask of additional detected thin clouds
    """
    # Convert to different color spaces for feature extraction
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    # Extract relevant channels
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    # Calculate local variance of L channel (texture feature)
    l_channel = lab[:,:,0]
    mean_l = cv2.GaussianBlur(l_channel, (15, 15), 0)
    local_variance = cv2.GaussianBlur((l_channel - mean_l)**2, (15, 15), 0)
    
    # Calculate red-blue difference normalized by intensity
    r = img_rgb[:,:,0].astype(float)
    g = img_rgb[:,:,1].astype(float)
    b = img_rgb[:,:,2].astype(float)
    intensity = (r + g + b) / 3.0
    rb_diff_norm = np.abs(r - b) / (intensity + 1e-6)
    
    # Key thin cloud feature: reduced blue intensity relative to red and green
    blue_deficit = ((r + g) / 2.0 - b) / (intensity + 1e-6)
    
    # Identify thin clouds based on multiple subtle features
    thin_cloud_mask = np.zeros_like(sky_mask)
    
    # Thin cloud features - multiple conditions with a scoring approach
    score_map = np.zeros_like(nrbr, dtype=float)
    
    # Add scores for each thin cloud indicator
    score_map += ((nrbr > -0.5) & (nrbr < -0.05)).astype(float) * 1.0  # NRBR in thin cloud range
    score_map += ((blue_deficit > 0.05) & (blue_deficit < 0.3)).astype(float) * 1.0  # Moderate blue deficit
    score_map += (local_variance > 20).astype(float) * 0.8  # Texture variance
    score_map += ((s_channel < 100) & (s_channel > 40)).astype(float) * 0.6  # Medium-low saturation
    score_map += ((v_channel > 150) & (v_channel < 220)).astype(float) * 0.5  # Medium brightness
    
    # Threshold the score map - areas with multiple matching features are likely thin clouds
    thin_cloud_mask[((score_map > 2.0) & (sky_mask > 0))] = 255
    
    # Clean up the mask
    thin_cloud_mask = cv2.morphologyEx(thin_cloud_mask, cv2.MORPH_OPEN, 
                                     np.ones((2, 2), np.uint8))
    thin_cloud_mask = cv2.morphologyEx(thin_cloud_mask, cv2.MORPH_CLOSE, 
                                     np.ones((3, 3), np.uint8))
    
    return thin_cloud_mask

def calculate_sun_obscuration_percentage(img_rgb, sun_position, clear_sky_img=None):
    """
    Calculate the percentage of sun obscuration by analyzing whiteness in sun mask region
    
    Args:
        img_rgb: RGB image to analyze
        sun_position: SunPosition object containing sun position mask
        clear_sky_img: Optional clear sky image for comparison
    
    Returns:
        float: Percentage of sun obscuration (0.0 = fully visible, 1.0 = fully obscured)
    """
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Extract channels
    s_channel = hsv[:,:,1]  # Saturation channel
    v_channel = hsv[:,:,2]  # Value channel (brightness)
    l_channel = lab[:,:,0]  # Lightness channel
    
    # Get only the sun mask region
    sun_mask = sun_position.mask.copy()
    
    if np.sum(sun_mask) == 0:
        return 1.0  # No sun region detected, consider fully obscured
    
    # Calculate statistics only in the sun mask region
    sun_brightness = np.mean(v_channel[sun_mask > 0])
    sun_saturation = np.mean(s_channel[sun_mask > 0])
    sun_lightness = np.mean(l_channel[sun_mask > 0])
    
    # Calculate whiteness score based on multiple criteria with stricter thresholds
    whiteness_scores = []
    
    # 1. Brightness-based whiteness (0-1, higher is whiter) - MORE STRICT
    # Require very high brightness (at least 220 out of 255)
    brightness_score = max(0.0, min(1.0, (sun_brightness - 220) / 35.0))  # 220-255 range
    whiteness_scores.append(brightness_score)
    
    # 2. Low saturation indicates whiteness (0-1, lower is whiter) - MORE STRICT
    # Require very low saturation (less than 30 out of 255)
    saturation_score = max(0.0, 1.0 - (sun_saturation / 30.0))  # 0-30 range
    whiteness_scores.append(saturation_score)
    
    # 3. High lightness indicates whiteness (0-1, higher is whiter) - MORE STRICT
    # Require very high lightness (at least 200 out of 255)
    lightness_score = max(0.0, min(1.0, (sun_lightness - 200) / 55.0))  # 200-255 range
    whiteness_scores.append(lightness_score)
    
    # 4. RGB balance - white should have balanced RGB values - MORE STRICT
    r = img_rgb[:,:,0].astype(float)
    g = img_rgb[:,:,1].astype(float)
    b = img_rgb[:,:,2].astype(float)
    
    sun_r = np.mean(r[sun_mask > 0])
    sun_g = np.mean(g[sun_mask > 0])
    sun_b = np.mean(b[sun_mask > 0])
    
    # Calculate RGB balance with stricter requirements
    # Require very small standard deviation (less than 15)
    rgb_std = np.std([sun_r, sun_g, sun_b])
    rgb_balance = max(0.0, 1.0 - (rgb_std / 15.0))  # 0-15 range
    whiteness_scores.append(rgb_balance)
    
    # 5. Calculate NRBR in sun region (negative values indicate blueness, positive indicate redness)
    nrbr_sun = calculate_nrbr(img_rgb)
    nrbr_sun_mean = np.mean(nrbr_sun[sun_mask > 0])
    
    # NRBR score: closer to 0 means more neutral/white - MORE STRICT
    # Require very small absolute NRBR value (less than 0.1)
    nrbr_score = max(0.0, 1.0 - (abs(nrbr_sun_mean) / 0.1))  # 0-0.1 range
    whiteness_scores.append(nrbr_score)
    
    # 6. ADDITIONAL: Check for very high individual RGB values
    # All RGB values should be very high (at least 200 each)
    min_rgb = min(sun_r, sun_g, sun_b)
    rgb_min_score = max(0.0, min(1.0, (min_rgb - 200) / 55.0))  # 200-255 range
    whiteness_scores.append(rgb_min_score)
    
    # Calculate overall whiteness score (average of all criteria)
    overall_whiteness = np.mean(whiteness_scores)
    
    # Convert whiteness to obscuration percentage
    # Higher whiteness = lower obscuration
    obscuration_percentage = 1.0 - overall_whiteness
    
    # If clear sky reference is available, use it for comparison
    if clear_sky_img is not None:
        clear_sky_hsv = cv2.cvtColor(clear_sky_img, cv2.COLOR_RGB2HSV)
        clear_sky_v = clear_sky_hsv[:,:,2]
        clear_sky_s = clear_sky_hsv[:,:,1]
        
        if np.sum(sun_mask) > 0:
            clear_sun_brightness = np.mean(clear_sky_v[sun_mask > 0])
            clear_sun_saturation = np.mean(clear_sky_s[sun_mask > 0])
            
            # Calculate brightness and saturation differences
            brightness_diff_ratio = max(0.0, (clear_sun_brightness - sun_brightness) / clear_sun_brightness)
            saturation_diff_ratio = max(0.0, (sun_saturation - clear_sun_saturation) / 255.0)
            
            # Adjust obscuration based on clear sky reference
            reference_adjustment = (brightness_diff_ratio + saturation_diff_ratio) / 2.0
            obscuration_percentage = min(1.0, obscuration_percentage + reference_adjustment * 0.3)
    
    # Ensure result is between 0 and 1
    obscuration_percentage = max(0.0, min(1.0, obscuration_percentage))
    
    return obscuration_percentage

def detect_transition_zones(nrbr_image, sky_mask):
    """
    Detect transition zones between clear sky and clouds based on NRBR gradients.
    
    Args:
        nrbr_image: NRBR values as numpy array
        sky_mask: Binary mask of sky region
        
    Returns:
        Binary mask of transition zones
    """
    # Calculate gradient magnitude of NRBR
    sobelx = cv2.Sobel(nrbr_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(nrbr_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient to 0-1 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # IMPROVED: More sensitive threshold to detect transitions
    transition_mask = np.zeros_like(sky_mask)
    transition_mask[(gradient_magnitude > 0.08) & (gradient_magnitude < 0.8) & (sky_mask > 0)] = 255
    
    # Dilate slightly to ensure coverage of transition areas
    kernel = np.ones((3, 3), np.uint8)
    transition_mask = cv2.dilate(transition_mask, kernel, iterations=1)
    
    return transition_mask

def protect_blue_sky_areas(img_rgb, nrbr, sky_mask, cloud_mask):
    """
    Identify and protect definite blue sky areas from being classified as clouds
    
    Args:
        img_rgb: RGB image
        nrbr: NRBR values
        sky_mask: Binary mask of sky region
        cloud_mask: Initial cloud mask to be refined
    
    Returns:
        Refined cloud mask with protected blue sky areas
    """
    # Convert to HSV 
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    # IMPROVED: Better blue sky detection criteria
    blue_sky_mask = np.zeros_like(sky_mask)
    blue_sky_criteria = (
        (h_channel >= 100) & (h_channel <= 140) &  # Blue hue range
        (s_channel > 80) &  # Higher saturation threshold to be more selective
        (v_channel > 160) &  # Medium to high brightness
        (nrbr < -0.45) &  # More negative NRBR values for definite clear blue sky
        (sky_mask > 0)  # Within sky region
    )
    blue_sky_mask[blue_sky_criteria] = 255
    
    # Apply morphological operations to clean the mask
    blue_sky_mask = cv2.morphologyEx(blue_sky_mask, cv2.MORPH_OPEN, 
                                   np.ones((3, 3), np.uint8))
    
    # Remove blue sky areas from cloud mask
    refined_cloud_mask = cloud_mask.copy()
    refined_cloud_mask[blue_sky_mask > 0] = 0
    
    return refined_cloud_mask, blue_sky_mask

def refine_cloud_edges(cloud_mask, transition_zones):
    """
    Apply edge refinement based on transition zones
    
    Args:
        cloud_mask: Binary cloud mask
        transition_zones: Binary mask of transition zones
        
    Returns:
        Refined cloud mask with less sensitivity at edges
    """
    # Create edge mask
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(cloud_mask, kernel, iterations=1)
    eroded = cv2.erode(cloud_mask, kernel, iterations=1)
    edge_mask = cv2.subtract(dilated, eroded)
    
    # In transition zones that are also edges, be more conservative
    conservative_mask = cloud_mask.copy()
    overlap_mask = (edge_mask > 0) & (transition_zones > 0)
    conservative_mask[overlap_mask] = 0
    
    # Smooth the result
    refined_mask = cv2.medianBlur(conservative_mask, 5)
    
    return refined_mask

def update_json_file(image_path, cloud_coverage_ratio, sun_obscuration_percentage, json_dir):
    """
    Update or create JSON file with cloud detection results while preserving irradiance values
    
    Args:
        image_path: Path to the processed image
        cloud_coverage_ratio: Float value of cloud coverage (0-1)
        sun_obscuration_percentage: Float value of sun obscuration (0-1)
        json_dir: Directory containing JSON files
    """
    # Extract date from image filename
    filename = os.path.basename(image_path)
    filename = os.path.splitext(filename)[0]  # Remove extension
    date = datetime.strptime(filename, '%Y%m%d_%H%M%S')
    
    # Determine JSON file path
    month_dir = os.path.join(json_dir, f"{date.month:02d}")
    json_file = os.path.join(month_dir, f"{date.day:02d}.json")
    
    # Create month directory if it doesn't exist
    os.makedirs(month_dir, exist_ok=True)
    
    # Load existing JSON data or create new list
    json_data = []
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading JSON file {json_file}: {e}")
            json_data = []
    
    # READ IRRADIANCE FROM SOURCE DIRECTORY (D:\2025\json)
    source_irradiance = None
    source_month_dir = os.path.join("D:\\2025\\json", f"{date.month:02d}")
    source_json_file = os.path.join(source_month_dir, f"{date.day:02d}.json")
    
    print(f"Looking for irradiance in source file: {source_json_file}")
    if os.path.exists(source_json_file):
        try:
            with open(source_json_file, 'r', encoding='utf-8') as f:
                source_data = [json.loads(line) for line in f if line.strip()]
            
            # Find entry with same timestamp that has irradiance
            for entry in source_data:
                if entry.get('now') == filename and 'irradiance' in entry:
                    source_irradiance = entry['irradiance']
                    print(f"Found irradiance value in source: {source_irradiance}")
                    break
        except Exception as e:
            print(f"Error reading source JSON file {source_json_file}: {e}")
    else:
        print(f"Source JSON file not found: {source_json_file}")
    
    # Find existing entry with irradiance value
    existing_irradiance = None
    print(f"Looking for timestamp: {filename}")
    print(f"JSON file being read: {json_file}")
    print(f"Total entries in file: {len(json_data)}")
    for entry in json_data:
        # Look for entry with same timestamp that has irradiance
        print(f"Checking entry: now='{entry.get('now')}', has_irradiance={'irradiance' in entry}")
        if entry.get('now') == filename and 'irradiance' in entry:
            existing_irradiance = entry['irradiance']
            print(f"Found existing irradiance value: {existing_irradiance}")
            break
    
    # Find and update matching entry or add new one
    updated = False
    for entry in json_data:
        if entry.get('image_path') == image_path:
            # Update cloud detection results
            entry['cloud_coverage'] = float(f"{cloud_coverage_ratio:.4f}")
            entry['sun_obscuration_percentage'] = float(f"{sun_obscuration_percentage:.4f}")
            
            # Add irradiance if it exists in the file
            if existing_irradiance is not None:
                entry['irradiance'] = existing_irradiance
                
            updated = True
            break
    
    # If no matching entry was found, create a new one
    if not updated:
        new_entry = {
            'image_path': image_path,
            'now': filename,
            'time': datetime.timestamp(date),
            'cloud_coverage': float(f"{cloud_coverage_ratio:.4f}"),
            'sun_obscuration_percentage': float(f"{sun_obscuration_percentage:.4f}")
        }
        
        # Add irradiance from source file if found
        if source_irradiance is not None:
            new_entry['irradiance'] = source_irradiance
            print(f"Added new entry for {image_path} with irradiance value from source: {source_irradiance}")
        else:
            print(f"Added new entry for {image_path} (no irradiance value found in source)")
            
        json_data.append(new_entry)
    
    # Save updated JSON data
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            for entry in json_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"Successfully updated {json_file} with {len(json_data)} entries")
    except Exception as e:
        print(f"Error saving JSON file {json_file}: {e}")

def calculate_cloud_coverage(image_path, timestamp=None, clear_sky_dir=None, sky_mask_dir=None, json_dir=None):
    """
    Detect clouds using enhanced method with sunshine correction and better thin cloud detection
    
    Args:
        image_path: Path to the fisheye image
        timestamp: Optional datetime object. If None, extracts from filename
        clear_sky_dir: Optional directory containing clear sky images
        sky_mask_dir: Directory containing pre-computed sky masks
        json_dir: Directory to store JSON files with results
    """
    # Load and preprocess the target image first to get its size
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    target_height, target_width = img.shape[:2]
    target_size = (target_height, target_width)
    
    # Load pre-computed sky mask
    if sky_mask_dir is None:
        sky_mask_dir = "D:\\Image\\test_result\\sky_detection\\masks"
    
    try:
        sky_mask, obstacle_mask, edge_mask, _ = load_sky_mask(sky_mask_dir)
        # Resize masks if needed
        if sky_mask.shape[:2] != target_size:
            sky_mask = cv2.resize(sky_mask, (target_width, target_height))
            obstacle_mask = cv2.resize(obstacle_mask, (target_width, target_height))
            edge_mask = cv2.resize(edge_mask, (target_width, target_height))
    except Exception as e:
        print(f"Error loading sky mask: {e}")
        print("Falling back to creating new sky mask...")
        sky_mask, obstacle_mask, edge_mask, sky_only = create_sky_mask(image_path, target_size=target_size)
    
    # Get timestamp from filename if not provided
    if timestamp is None:
        filename = os.path.basename(image_path)
        filename = os.path.splitext(filename)[0]
        base_timestamp = datetime.strptime(filename, '%Y%m%d_%H%M%S')
        # Add 8 hours since 0 hour represents 8:00 AM
        timestamp = base_timestamp.replace(hour=base_timestamp.hour + 8)
    
    # Get sun position for circumsolar area handling
    image_center = (target_width // 2, target_height // 2)
    calculator = SunPositionCalculator(
        image_size=target_size,
        image_center=image_center,
        image_radius=min(target_width, target_height) // 2 - 10
    )
    sun_position = calculator.calculate_position(timestamp)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ENHANCEMENT: Apply contrast enhancement before cloud detection
    enhanced_img = enhance_contrast(img_rgb, alpha=1.3, beta=10)
    
    # Calculate NRBR for enhanced image
    nrbr = calculate_nrbr(enhanced_img)
    
    # Initialize cloud mask
    cloud_mask = np.zeros_like(sky_mask)
    
    # IMPROVED: Adjusted thresholds for better balance
    NRBR_CLOUD_THRESHOLD = -0.28  # More sensitive to clouds
    DELTA_NRBR_THRESHOLD = 0.16   # More sensitive to changes
    CLEAR_SKY_THRESHOLD = 0.02    # More sensitive to initial cloudiness
    VERY_CLOUDY_THRESHOLD = 0.35  # Adjusted for better balance
    
    # Get appropriate clear sky image from library
    clear_sky_path = None
    clear_sky_img = None
    if clear_sky_dir:
        clear_sky_path = get_clear_sky_image(timestamp, clear_sky_dir)
        if clear_sky_path and os.path.exists(clear_sky_path):
            clear_sky_img = cv2.imread(clear_sky_path)
            if clear_sky_img is not None:
                clear_sky_img = cv2.resize(clear_sky_img, (target_width, target_height))
                clear_sky_img = cv2.cvtColor(clear_sky_img, cv2.COLOR_BGR2RGB)
    
    # Detect transition zones for edge refinement
    transition_zones = detect_transition_zones(nrbr, sky_mask)
    
    # Basic cloud detection using NRBR and/or clear sky library
    if clear_sky_path and os.path.exists(clear_sky_path):
        # Load and resize clear sky image to match target image size
        cs_img = cv2.imread(clear_sky_path)
        if cs_img is None:
            raise ValueError(f"Could not read clear sky image: {clear_sky_path}")
            
        cs_img = cv2.resize(cs_img, (target_width, target_height))
        cs_rgb = cv2.cvtColor(cs_img, cv2.COLOR_BGR2RGB)
        nrbr_cs = calculate_nrbr(cs_rgb)
        
        # Calculate NRBR difference
        delta_nrbr = np.abs(nrbr - nrbr_cs)
        
        # Simplified confidence map
        confidence_map = np.ones_like(nrbr, dtype=float)
        confidence_map[transition_zones > 0] = 0.6  # Less penalty in transition zones
        
        # Simplified threshold map
        delta_threshold_map = np.ones_like(nrbr, dtype=float) * DELTA_NRBR_THRESHOLD
        delta_threshold_map[transition_zones > 0] = DELTA_NRBR_THRESHOLD * 1.1  # 10% higher threshold in transition zones
        
        # Get initial cloudiness estimate
        valid_sky = (sky_mask > 0)
        cloudiness = np.sum((delta_nrbr >= DELTA_NRBR_THRESHOLD) & valid_sky) / np.sum(valid_sky)
        
        # Simplified detection logic
        if cloudiness < CLEAR_SKY_THRESHOLD:
            # Clear sky - use CSL method only
            cloud_mask[(delta_nrbr >= delta_threshold_map) & valid_sky] = 255
        elif cloudiness > VERY_CLOUDY_THRESHOLD:
            # Very cloudy - use NRBR method only
            cloud_mask[(nrbr > NRBR_CLOUD_THRESHOLD) & valid_sky] = 255
        else:
            # Partly cloudy - use combined method
            outside_sun = (sun_position.mask == 0)
            inside_sun = (sun_position.mask > 0)
            
            # Use NRBR outside sun region
            cloud_mask[(nrbr > NRBR_CLOUD_THRESHOLD) & valid_sky & outside_sun] = 255
            
            # Use CSL method in sun region
            cloud_mask[(delta_nrbr >= delta_threshold_map) & valid_sky & inside_sun] = 255
    else:
        # Fallback to NRBR-only method if no clear sky image
        cloud_mask[(nrbr > NRBR_CLOUD_THRESHOLD) & (sky_mask > 0)] = 255
    
    # IMPROVED: Enhanced thin cloud detection using enhanced image
    thin_cloud_mask = enhance_thin_cloud_detection(enhanced_img, nrbr, sky_mask)
    
    # ADDED: Additional thin cloud detection with multi-feature approach using enhanced image
    additional_thin_clouds = detect_additional_thin_clouds(enhanced_img, nrbr, sky_mask)
    
    # Combine all detected clouds
    cloud_mask = cv2.bitwise_or(cloud_mask, thin_cloud_mask)
    cloud_mask = cv2.bitwise_or(cloud_mask, additional_thin_clouds)
    
    # CRITICAL: Detect sunshine area AFTER all cloud detection steps using enhanced image
    sunshine_mask = detect_sunshine_area(enhanced_img, sun_position)
    
    # CRITICAL: Remove sunshine area from cloud mask to prevent false positives
    cloud_mask[sunshine_mask > 0] = 0
    
    # Protect definite blue sky areas and get the blue sky mask using enhanced image
    cloud_mask, blue_sky_mask = protect_blue_sky_areas(enhanced_img, nrbr, sky_mask, cloud_mask)
    
    # Calculate sun obscuration percentage (use enhanced method)
    sun_obscuration_percentage = calculate_sun_obscuration_percentage(enhanced_img, sun_position, clear_sky_img)
    
    # Final morphological operations for clean result
    kernel = np.ones((3, 3), np.uint8)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate cloud coverage ratio based on sun obscuration percentage
    if sun_obscuration_percentage < 0.5:
        # When sun is mostly visible (obscuration < 50%), exclude sun region from calculations
        valid_sky = (sky_mask > 0) & (sun_position.mask == 0)  # Only sky area excluding sun region
        valid_pixels = np.sum(valid_sky)
        cloud_pixels = np.sum(cloud_mask > 0 & valid_sky)  # Only count clouds in valid sky area
    else:
        # When sun is mostly obscured (obscuration >= 50%), include all sky regions
        valid_sky = (sky_mask > 0)
        valid_pixels = np.sum(valid_sky)
        cloud_pixels = np.sum(cloud_mask > 0 & valid_sky)
    
    cloud_coverage_ratio = cloud_pixels / valid_pixels if valid_pixels > 0 else 0
    
    # Update JSON file if directory is provided
    if json_dir:
        update_json_file(image_path, cloud_coverage_ratio, sun_obscuration_percentage, json_dir)
    
    # Create visualization
    visualization = img_rgb.copy()
    visualization_overlay = np.zeros_like(img_rgb)
    
    # Create distinct masks for visualization
    if sun_obscuration_percentage < 0.5:
        # When sun is mostly visible (obscuration < 50%), remove sun region from all masks
        sun_mask = sun_position.mask > 0
        thick_cloud_mask = cv2.subtract(cloud_mask, cv2.bitwise_or(thin_cloud_mask, additional_thin_clouds))
        thin_cloud_combined = cv2.bitwise_or(thin_cloud_mask, additional_thin_clouds)
        
        # Remove sun region from all masks
        thick_cloud_mask[sun_mask] = 0
        thin_cloud_combined[sun_mask] = 0
        blue_sky_mask[sun_mask] = 0
        sunshine_mask[sun_mask] = 0
    else:
        thick_cloud_mask = cv2.subtract(cloud_mask, cv2.bitwise_or(thin_cloud_mask, additional_thin_clouds))
        thin_cloud_combined = cv2.bitwise_or(thin_cloud_mask, additional_thin_clouds)
    
    # Mark different areas with colors in RGB format
    visualization_overlay[thick_cloud_mask > 0] = [0, 255, 0]  # Green for thick clouds
    visualization_overlay[thin_cloud_combined > 0] = [255, 255, 0]  # Yellow for thin clouds
    visualization_overlay[blue_sky_mask > 0] = [0, 0, 255]     # Blue for protected blue sky
    if sun_obscuration_percentage >= 0.5:
        visualization_overlay[sunshine_mask > 0] = [255, 165, 0]   # Orange for sunshine area only when sun is mostly obscured
    
    # Add semi-transparent overlay
    alpha = 0.4
    mask = (thick_cloud_mask > 0) | (thin_cloud_combined > 0) | (blue_sky_mask > 0) | (sunshine_mask > 0)
    visualization = np.where(
        np.repeat(mask[:, :, np.newaxis], 3, axis=2),
        visualization * (1 - alpha) + visualization_overlay * alpha,
        visualization
    ).astype(np.uint8)
    
    # Mark sun region
    if np.sum(sun_position.mask) > 0:
        sun_center = np.argwhere(sun_position.mask > 0).mean(axis=0).astype(int)
        
        sun_radius = int(np.sqrt(np.sum(sun_position.mask) / np.pi))
        cv2.circle(visualization, (sun_center[1], sun_center[0]), sun_radius, [255, 0, 0], 2)
    
        # Add text with cloud coverage and sun obscuration
    cv2.putText(
        visualization,
        f'Cloud Coverage: {cloud_coverage_ratio:.1%}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    cv2.putText(
        visualization,
        f'Sun Obscuration: {sun_obscuration_percentage:.1%}',
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # Create analysis plot
    plt.figure(figsize=(24, 5))
    
    plt.subplot(1, 6, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(enhanced_img)
    plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(sky_mask, cmap='gray')
    plt.title('Sky Mask')
    plt.axis('off')
    
    plt.subplot(1, 6, 4)
    plt.imshow(nrbr, cmap='RdBu')
    plt.title('NRBR')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 6, 5)
    plt.imshow(cloud_mask, cmap='gray')
    plt.title('Cloud Mask')
    plt.axis('off')
    
    plt.subplot(1, 6, 6)
    plt.imshow(visualization)
    plt.title('Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cloud_detection_analysis.png')
    plt.close()

    return cloud_coverage_ratio, visualization, sun_obscuration_percentage

def process_single_image(image_path, clear_sky_dir=None, sky_mask_dir=None, json_dir=None):
    """
    Process a single image with cloud detection
    """
    try:
        coverage, vis, sun_obscuration_percentage = calculate_cloud_coverage(
            image_path,
            clear_sky_dir=clear_sky_dir,
            sky_mask_dir=sky_mask_dir,
            json_dir=json_dir
        )
        
        # Save visualization only if successful
        if vis is not None:
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            # Save visualization
            vis_dir = os.path.join(os.path.dirname(json_dir), "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{base_name}_result.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return {
            'filename': os.path.basename(image_path),
            'cloud_coverage': coverage,
            'sun_obscuration_percentage': sun_obscuration_percentage,
            'timestamp': datetime.strptime(os.path.splitext(os.path.basename(image_path))[0], '%Y%m%d_%H%M%S'),
            'success': True
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {
            'filename': os.path.basename(image_path),
            'error': str(e),
            'success': False
        }

def main():
    # Source directory containing images to test
    source_dir = "D:\\Image"
    
    # Create output directory for test results
    output_dir = "D:\\Image\\test_result"
    os.makedirs(output_dir, exist_ok=True)
    
    # Directory for JSON files
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    
    # Create subdirectories for different types of results
    vis_dir = os.path.join(output_dir, "visualizations")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get list of all jpg images in the source directory
    image_files = glob.glob(os.path.join(source_dir, "*.jpg"))
    image_files.sort()  # Sort files to ensure consistent order
    
    # Specify the directory containing clear sky images
    clear_sky_dir = "D:\\Image\\clear-sky-lib"
    
    # Specify the directory containing pre-computed sky masks
    sky_mask_dir = "D:\\Image\\test_result\\sky_detection\\masks"
    
    print(f"Processing {len(image_files)} images...")
    print(f"Using clear sky library: {clear_sky_dir}")
    print(f"Using pre-computed sky masks: {sky_mask_dir}")
    print(f"Using JSON directory: {json_dir}")
    
    # Calculate number of processes to use (leave one core free for system)
    num_processes = min(6, max(1, cpu_count() - 1))  # Limit to maximum 6 cores
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_image,
                clear_sky_dir=clear_sky_dir,
                sky_mask_dir=sky_mask_dir,
        json_dir=json_dir
    )
    
    # Process images in parallel
    results = []
    with Pool(processes=num_processes) as pool:
        # Process images in chunks for better progress tracking
        chunk_size = max(1, len(image_files) // (num_processes * 4))
        for i, result in enumerate(pool.imap(process_func, image_files, chunksize=chunk_size)):
            if result['success']:
                results.append(result)
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        # Create summary DataFrame
        df = pd.DataFrame(successful_results)
    
        # Save summary to CSV
        summary_path = os.path.join(output_dir, "test_results_summary.csv")
        df.to_csv(summary_path, index=False)
        
        # Generate summary statistics
        print("\nTest Results Summary:")
        print(f"Total images processed successfully: {len(successful_results)}/{len(image_files)}")
        print(f"Average cloud coverage: {df['cloud_coverage'].mean():.1%}")
        print(f"Sun obscured in {df['sun_obscured'].sum()} images")
        print(f"\nResults saved in: {output_dir}")
        print(f"- Visualizations: {vis_dir}")
        print(f"- Analysis plots: {analysis_dir}")
        print(f"- Summary CSV: {summary_path}")
    else:
        print("No images were processed successfully")

if __name__ == "__main__":
    main()