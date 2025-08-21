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

class CloudDetectionDemo:
    def __init__(self, output_dir="D:\\Image\\test_result\\cloud_detection_demo"):
        """Initialize the cloud detection demo with visualization capabilities"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different visualization steps
        self.steps = [
            "01_input",
            "02_sky_mask",
            "03_sun_position",
            "04_nrbr",
            "05_thin_clouds",
            "06_sunshine",
            "07_final_result"
        ]
        
        for step in self.steps:
            os.makedirs(os.path.join(output_dir, step), exist_ok=True)

    def visualize_input_image(self, img_rgb, step_name="01_input"):
        """Visualize input image and its color channels"""
        print("\nStep 1: Input Image Analysis")
        print("-" * 50)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original RGB image
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Original RGB Image')
        axs[0, 0].axis('off')
        
        # Individual color channels
        axs[0, 1].imshow(img_rgb[:,:,0], cmap='Reds')
        axs[0, 1].set_title('Red Channel')
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(img_rgb[:,:,1], cmap='Greens')
        axs[1, 0].set_title('Green Channel')
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(img_rgb[:,:,2], cmap='Blues')
        axs[1, 1].set_title('Blue Channel')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "color_channels.png"))
        plt.close()
        
        # Create HSV visualization
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(hsv[:,:,0], cmap='hsv')
        axs[0].set_title('Hue')
        axs[0].axis('off')
        
        axs[1].imshow(hsv[:,:,1], cmap='gray')
        axs[1].set_title('Saturation')
        axs[1].axis('off')
        
        axs[2].imshow(hsv[:,:,2], cmap='gray')
        axs[2].set_title('Value')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "hsv_channels.png"))
        plt.close()

    def visualize_sky_mask(self, img_rgb, sky_mask, obstacle_mask, edge_mask, step_name="02_sky_mask"):
        """Visualize sky mask creation process"""
        print("\nStep 2: Sky Mask Generation")
        print("-" * 50)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image with sky mask overlay
        overlay = img_rgb.copy()
        overlay[sky_mask > 0] = [0, 0, 255]  # Blue for sky
        overlay = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        
        axs[0, 0].imshow(overlay)
        axs[0, 0].set_title('Sky Mask Overlay')
        axs[0, 0].axis('off')
        
        # Sky mask
        axs[0, 1].imshow(sky_mask, cmap='gray')
        axs[0, 1].set_title('Sky Mask')
        axs[0, 1].axis('off')
        
        # Obstacle mask
        axs[1, 0].imshow(obstacle_mask, cmap='gray')
        axs[1, 0].set_title('Obstacle Mask')
        axs[1, 0].axis('off')
        
        # Edge mask
        axs[1, 1].imshow(edge_mask, cmap='gray')
        axs[1, 1].set_title('Edge Mask')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "sky_mask_analysis.png"))
        plt.close()

    def visualize_sun_position(self, img_rgb, sun_position, step_name="03_sun_position"):
        """Visualize sun position detection"""
        print("\nStep 3: Sun Position Detection")
        print("-" * 50)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image with sun position
        ax1.imshow(img_rgb)
        if np.sum(sun_position.mask) > 0:
            sun_center = np.argwhere(sun_position.mask > 0).mean(axis=0).astype(int)
            sun_radius = int(np.sqrt(np.sum(sun_position.mask) / np.pi))
            circle = plt.Circle((sun_center[1], sun_center[0]), sun_radius, 
                              color='yellow', alpha=0.5)
            ax1.add_artist(circle)
            ax1.plot(sun_center[1], sun_center[0], 'r+', markersize=10)
        ax1.set_title('Sun Position on Image')
        ax1.axis('off')
        
        # Sun mask
        ax2.imshow(sun_position.mask, cmap='gray')
        ax2.set_title('Sun Mask')
        ax2.axis('off')
        
        # Add text with sun position information
        info_text = (
            f"Sun Position (x, y): ({sun_position.x}, {sun_position.y})\n"
            f"Azimuth: {sun_position.azimuth:.2f}°\n"
            f"Zenith: {sun_position.zenith:.2f}°"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "sun_position.png"))
        plt.close()

    def visualize_nrbr(self, img_rgb, nrbr, step_name="04_nrbr"):
        """Visualize NRBR calculation and analysis"""
        print("\nStep 4: NRBR Analysis")
        print("-" * 50)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # NRBR visualization
        nrbr_vis = axs[0, 1].imshow(nrbr, cmap='RdBu')
        axs[0, 1].set_title('NRBR Values')
        axs[0, 1].axis('off')
        plt.colorbar(nrbr_vis, ax=axs[0, 1])
        
        # NRBR histogram
        axs[1, 0].hist(nrbr.ravel(), bins=50, range=(-1, 1))
        axs[1, 0].set_title('NRBR Histogram')
        axs[1, 0].set_xlabel('NRBR Value')
        axs[1, 0].set_ylabel('Frequency')
        
        # NRBR thresholds visualization
        threshold_vis = np.zeros_like(nrbr)
        threshold_vis[nrbr > -0.28] = 1  # Cloud threshold
        threshold_vis[nrbr < -0.45] = 2  # Clear sky threshold
        axs[1, 1].imshow(threshold_vis, cmap='viridis')
        axs[1, 1].set_title('NRBR Thresholds\nBlue: Clear Sky')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "nrbr_analysis.png"))
        plt.close()

    def visualize_thin_clouds(self, img_rgb, thin_cloud_mask, additional_thin_clouds, 
                            local_contrast, gradient_magnitude, step_name="05_thin_clouds"):
        """Visualize thin cloud detection process"""
        print("\nStep 5: Thin Cloud Detection")
        print("-" * 50)
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # Primary thin cloud detection
        axs[0, 1].imshow(thin_cloud_mask, cmap='gray')
        axs[0, 1].set_title('Primary Thin Cloud Mask')
        axs[0, 1].axis('off')
        
        # Additional thin cloud detection
        axs[0, 2].imshow(additional_thin_clouds, cmap='gray')
        axs[0, 2].set_title('Additional Thin Clouds')
        axs[0, 2].axis('off')
        
        # Local contrast
        contrast_vis = axs[1, 0].imshow(local_contrast, cmap='viridis')
        axs[1, 0].set_title('Local Contrast')
        axs[1, 0].axis('off')
        plt.colorbar(contrast_vis, ax=axs[1, 0])
        
        # Gradient magnitude
        grad_vis = axs[1, 1].imshow(gradient_magnitude, cmap='viridis')
        axs[1, 1].set_title('Gradient Magnitude')
        axs[1, 1].axis('off')
        plt.colorbar(grad_vis, ax=axs[1, 1])
        
        # Combined thin clouds
        combined = cv2.bitwise_or(thin_cloud_mask, additional_thin_clouds)
        axs[1, 2].imshow(combined, cmap='gray')
        axs[1, 2].set_title('Combined Thin Clouds')
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "thin_cloud_detection.png"))
        plt.close()

    def visualize_sunshine(self, img_rgb, sunshine_mask, sun_position, step_name="06_sunshine"):
        """Visualize sunshine area detection"""
        print("\nStep 6: Sunshine Area Detection")
        print("-" * 50)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image with sun position
        axs[0, 0].imshow(img_rgb)
        if np.sum(sun_position.mask) > 0:
            sun_center = np.argwhere(sun_position.mask > 0).mean(axis=0).astype(int)
            sun_radius = int(np.sqrt(np.sum(sun_position.mask) / np.pi))
            circle = plt.Circle((sun_center[1], sun_center[0]), sun_radius, 
                              color='yellow', alpha=0.5)
            axs[0, 0].add_artist(circle)
        axs[0, 0].set_title('Original with Sun Position')
        axs[0, 0].axis('off')
        
        # Sunshine mask
        axs[0, 1].imshow(sunshine_mask, cmap='gray')
        axs[0, 1].set_title('Sunshine Mask')
        axs[0, 1].axis('off')
        
        # HSV analysis of sunshine area
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sunshine_pixels = hsv[sunshine_mask > 0]
        
        if len(sunshine_pixels) > 0:
            # Brightness histogram
            axs[1, 0].hist(sunshine_pixels[:,2], bins=50, range=(0, 255))
            axs[1, 0].set_title('Brightness Distribution\nin Sunshine Area')
            axs[1, 0].set_xlabel('Brightness (V)')
            axs[1, 0].set_ylabel('Frequency')
            
            # Saturation histogram
            axs[1, 1].hist(sunshine_pixels[:,1], bins=50, range=(0, 255))
            axs[1, 1].set_title('Saturation Distribution\nin Sunshine Area')
            axs[1, 1].set_xlabel('Saturation (S)')
            axs[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "sunshine_detection.png"))
        plt.close()

    def visualize_final_result(self, img_rgb, cloud_mask, thin_cloud_mask, sunshine_mask, 
                             blue_sky_mask, sun_obscuration_percentage, step_name="07_final_result"):
        """Visualize final cloud detection result"""
        print("\nStep 7: Final Results")
        print("-" * 50)
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # Cloud mask
        axs[0, 1].imshow(cloud_mask, cmap='gray')
        axs[0, 1].set_title('Thick Cloud Mask')
        axs[0, 1].axis('off')
        
        # Thin cloud mask
        axs[0, 2].imshow(thin_cloud_mask, cmap='gray')
        axs[0, 2].set_title('Thin Cloud Mask')
        axs[0, 2].axis('off')
        
        # Blue sky mask
        axs[1, 0].imshow(blue_sky_mask, cmap='gray')
        axs[1, 0].set_title('Blue Sky Mask')
        axs[1, 0].axis('off')
        
        # Sunshine mask
        axs[1, 1].imshow(sunshine_mask, cmap='gray')
        axs[1, 1].set_title('Sunshine Mask')
        axs[1, 1].axis('off')
        
        # Final overlay
        overlay = img_rgb.copy()
        overlay[cloud_mask > 0] = [0, 255, 0]      # Green for thick clouds
        overlay[thin_cloud_mask > 0] = [255, 255, 0]  # Yellow for thin clouds
        overlay[blue_sky_mask > 0] = [0, 0, 255]   # Blue for clear sky
        overlay[sunshine_mask > 0] = [255, 165, 0]  # Orange for sunshine
        overlay = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        
        axs[1, 2].imshow(overlay)
        axs[1, 2].set_title('Final Result Overlay')
        axs[1, 2].axis('off')
        
        # Add text with analysis results
        valid_sky = (cloud_mask > 0) | (thin_cloud_mask > 0) | (blue_sky_mask > 0)
        cloud_coverage = np.sum((cloud_mask > 0) | (thin_cloud_mask > 0)) / np.sum(valid_sky) if np.sum(valid_sky) > 0 else 0
        
        info_text = (
            f"Analysis Results:\n"
            f"Cloud Coverage: {cloud_coverage:.1%}\n"
            f"Sun Obscuration: {sun_obscuration_percentage:.1%}\n"
            f"Thick Cloud Area: {np.sum(cloud_mask > 0) / np.sum(valid_sky):.1%}\n"
            f"Thin Cloud Area: {np.sum(thin_cloud_mask > 0) / np.sum(valid_sky):.1%}\n"
            f"Clear Sky Area: {np.sum(blue_sky_mask > 0) / np.sum(valid_sky):.1%}"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, step_name, "final_result.png"))
        plt.close()
        
        return cloud_coverage

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
    # Initialize visualization demo
    demo = CloudDetectionDemo()
    
    # Load and preprocess the target image first to get its size
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    target_height, target_width = img.shape[:2]
    target_size = (target_height, target_width)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Input Image Analysis
    demo.visualize_input_image(img_rgb)
    
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
    
    # Step 2: Sky Mask Visualization
    demo.visualize_sky_mask(img_rgb, sky_mask, obstacle_mask, edge_mask)
    
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
    
    # Step 3: Sun Position Visualization
    demo.visualize_sun_position(img_rgb, sun_position)
    
    # ENHANCEMENT: Apply contrast enhancement before cloud detection
    enhanced_img = enhance_contrast(img_rgb, alpha=1.3, beta=10)
    
    # Calculate NRBR for enhanced image
    nrbr = calculate_nrbr(enhanced_img)
    
    # Step 4: NRBR Analysis Visualization
    demo.visualize_nrbr(img_rgb, nrbr)
    
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
    
    # Calculate local contrast and gradient magnitude for visualization
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
    local_contrast = cv2.GaussianBlur(gray, (0, 0), 2) - cv2.GaussianBlur(gray, (0, 0), 16)
    sobelx = cv2.Sobel(nrbr, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(nrbr, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Step 5: Thin Cloud Detection Visualization
    demo.visualize_thin_clouds(enhanced_img, thin_cloud_mask, additional_thin_clouds, 
                             local_contrast, gradient_magnitude)
    
    # Combine all detected clouds
    cloud_mask = cv2.bitwise_or(cloud_mask, thin_cloud_mask)
    cloud_mask = cv2.bitwise_or(cloud_mask, additional_thin_clouds)
    
    # CRITICAL: Detect sunshine area AFTER all cloud detection steps using enhanced image
    sunshine_mask = detect_sunshine_area(enhanced_img, sun_position)
    
    # Step 6: Sunshine Area Detection Visualization
    demo.visualize_sunshine(enhanced_img, sunshine_mask, sun_position)
    
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
    
    # Step 7: Final Result Visualization
    demo.visualize_final_result(img_rgb, thick_cloud_mask, thin_cloud_combined, 
                              sunshine_mask, blue_sky_mask, sun_obscuration_percentage)
    
    return cloud_coverage_ratio, visualization, sun_obscuration_percentage

def enhance_thin_cloud_detection(img_rgb, nrbr, sky_mask):
    """
    Enhanced detection of thin clouds using multiple features
    
    Args:
        img_rgb: RGB image array
        nrbr: NRBR values array
        sky_mask: Binary mask of sky region
        
    Returns:
        Binary mask of detected thin clouds
    """
    # Convert to HSV for additional color analysis
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Calculate local contrast
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    local_contrast = cv2.GaussianBlur(gray, (0, 0), 2) - cv2.GaussianBlur(gray, (0, 0), 16)
    
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(nrbr, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(nrbr, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient to 0-1 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # Thin cloud criteria
    thin_cloud_mask = np.zeros_like(sky_mask)
    thin_cloud_criteria = (
        (-0.28 <= nrbr) & (nrbr <= -0.15) &  # Intermediate NRBR values
        (gradient_magnitude > 0.05) &  # Some texture present
        (hsv[:,:,1] < 100) &  # Lower saturation
        (hsv[:,:,2] > 180) &  # Higher brightness
        (sky_mask > 0)  # Within sky region
    )
    thin_cloud_mask[thin_cloud_criteria] = 255
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    thin_cloud_mask = cv2.morphologyEx(thin_cloud_mask, cv2.MORPH_OPEN, kernel)
    thin_cloud_mask = cv2.morphologyEx(thin_cloud_mask, cv2.MORPH_CLOSE, kernel)
    
    return thin_cloud_mask

def detect_additional_thin_clouds(img_rgb, nrbr, sky_mask):
    """
    Detect additional thin clouds using multi-feature approach
    
    Args:
        img_rgb: RGB image array
        nrbr: NRBR values array
        sky_mask: Binary mask of sky region
        
    Returns:
        Binary mask of additional detected thin clouds
    """
    # Convert to HSV and LAB color spaces
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Calculate local standard deviation (texture)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Convert boolean mask to uint8 for OpenCV
    mask_uint8 = np.uint8(sky_mask > 0) * 255
    mean, stddev = cv2.meanStdDev(gray, mask=mask_uint8)
    local_std = cv2.GaussianBlur(gray, (15, 15), 0) - cv2.GaussianBlur(gray, (31, 31), 0)
    
    # Additional thin cloud criteria
    additional_mask = np.zeros_like(sky_mask)
    thin_cloud_criteria = (
        (-0.2 <= nrbr) & (nrbr <= -0.1) &  # Slightly different NRBR range
        (hsv[:,:,1] < 80) &  # Very low saturation
        (lab[:,:,0] > 150) &  # High luminance
        (np.abs(local_std) > 3) &  # Some texture variation
        (sky_mask > 0)  # Within sky region
    )
    additional_mask[thin_cloud_criteria] = 255
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    additional_mask = cv2.morphologyEx(additional_mask, cv2.MORPH_OPEN, kernel)
    
    return additional_mask

def detect_sunshine_area(img_rgb, sun_position):
    """
    Detect sunshine area around the sun
    
    Args:
        img_rgb: RGB image array
        sun_position: SunPosition object with mask
        
    Returns:
        Binary mask of detected sunshine area
    """
    # Convert to HSV for better brightness analysis
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Get sun center and radius
    if np.sum(sun_position.mask) > 0:
        sun_center = np.argwhere(sun_position.mask > 0).mean(axis=0).astype(int)
        sun_radius = int(np.sqrt(np.sum(sun_position.mask) / np.pi))
        
        # Create distance map from sun center
        y, x = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        dist_from_sun = np.sqrt((x - sun_center[1])**2 + (y - sun_center[0])**2)
        
        # Create sunshine mask based on distance and brightness
        sunshine_mask = np.zeros_like(sun_position.mask)
        sunshine_criteria = (
            (dist_from_sun <= sun_radius * 2.5) &  # Within extended sun radius
            (hsv[:,:,1] < 60) &  # Low saturation
            (hsv[:,:,2] > 200)  # High brightness
        )
        sunshine_mask[sunshine_criteria] = 255
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        sunshine_mask = cv2.morphologyEx(sunshine_mask, cv2.MORPH_CLOSE, kernel)
        
        return sunshine_mask
    
    return np.zeros_like(img_rgb[:,:,0])

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
    # Test parameters
    image_path = "D:\\Image\\20250101_002700.jpg"
    output_dir = "D:\\Image\\test_result\\cloud_detection_demo"
    
    # Create output directory for test results
    os.makedirs(output_dir, exist_ok=True)
    
    # Directory for JSON files
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    
    # Specify the directory containing clear sky images
    clear_sky_dir = "D:\\Image\\clear-sky-lib"
    
    # Specify the directory containing pre-computed sky masks
    sky_mask_dir = "D:\\Image\\test_result\\sky_detection\\masks"
    
    print(f"Starting Cloud Detection Demo")
    print(f"{'='*50}")
    print(f"Input Image: {image_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Using clear sky library: {clear_sky_dir}")
    print(f"Using pre-computed sky masks: {sky_mask_dir}")
    
    try:
        # Process the image
        cloud_coverage, visualization, sun_obscuration_percentage = calculate_cloud_coverage(
            image_path,
            clear_sky_dir=clear_sky_dir,
            sky_mask_dir=sky_mask_dir,
            json_dir=json_dir
        )
        
        # Save visualization
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        vis_path = os.path.join(vis_dir, f"{base_name}_result.png")
        cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        print("\nResults:")
        print(f"Cloud Coverage: {cloud_coverage:.1%}")
        print(f"Sun Obscuration: {sun_obscuration_percentage:.1%}")
        print(f"\nResults saved in: {output_dir}")
        print(f"- Visualization: {vis_path}")
        print(f"- Analysis plot: cloud_detection_analysis.png")
        
    except Exception as e:
        print(f"Error processing image: {e}")

def main_batch():
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
        print(f"Average sun obscuration: {df['sun_obscuration_percentage'].mean():.1%}")
        print(f"\nResults saved in: {output_dir}")
        print(f"- Visualizations: {vis_dir}")
        print(f"- Analysis plots: {analysis_dir}")
        print(f"- Summary CSV: {summary_path}")
    else:
        print("No images were processed successfully")

if __name__ == "__main__":
    main() 