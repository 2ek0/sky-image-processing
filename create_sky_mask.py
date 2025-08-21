import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

def create_sky_mask(image_path, target_size=None, save_results=False, output_dir=None):
    """
    Create sky mask for a fisheye image with improved building structure detection
    
    Args:
        image_path: Path to the fisheye image
        target_size: Optional tuple (height, width) to resize image to
        save_results: Whether to save visualization results
        output_dir: Directory to save results if save_results is True
    
    Returns:
        tuple: (sky_mask, obstacle_mask, edge_mask, sky_only)
    """
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image file")
        
    # Resize if target size is specified
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a circular mask for the fisheye view
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y) - 110  # Slightly smaller than the circle to avoid the edge
    
    # Create a black mask with a white circle
    circle_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
    
    # ===== BUILDING STRUCTURE DETECTION =====
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply contrast enhancement to make building structures more visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply adaptive thresholding to detect building edges
    binary = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    
    # Apply Canny edge detection with optimized thresholds for building structures
    edges = cv2.Canny(blurred, 30, 90)
    
    # Only keep edges within the fisheye circle
    masked_edges = cv2.bitwise_and(edges, edges, mask=circle_mask)
    
    # Apply morphological operations to connect nearby edges
    kernel_line_h = np.ones((1, 7), np.uint8)  # Horizontal line kernel
    kernel_line_v = np.ones((7, 1), np.uint8)  # Vertical line kernel
    
    # Dilate edges to connect gaps
    dilated_h = cv2.dilate(masked_edges, kernel_line_h, iterations=1)
    dilated_v = cv2.dilate(masked_edges, kernel_line_v, iterations=1)
    dilated_edges = cv2.bitwise_or(dilated_h, dilated_v)
    
    # Find contours to identify building structures
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a solid building mask
    building_mask = np.zeros_like(dilated_edges)
    
    # Fill all the contours to create solid building areas
    cv2.drawContours(building_mask, contours, -1, 255, -1)  # -1 fills the contours
    
    # Additional morphological operations to fill any small gaps between building components
    kernel_close = np.ones((7, 7), np.uint8)
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Create a mask for the sky (primarily blue regions)
    # Convert to HSV for better color segmentation
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Define range for blue/sky color in HSV
    lower_sky = np.array([100, 80, 80])
    upper_sky = np.array([120, 255, 255])
    
    # Create mask for blue/sky colors
    sky_mask = cv2.inRange(img_hsv, lower_sky, upper_sky)
    
    # Create a mask for bright regions (including sun)
    _, bright_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Create a central bright region mask
    central_radius = (radius // 2) + 140  # Adjust this value to control the size of the central region
    central_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(central_mask, (center_x, center_y), central_radius, 255, -1)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(sky_mask, bright_mask)
    
    # Apply the circle mask to keep only the fisheye region
    final_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=circle_mask)
    
    # Fill holes and remove noise
    kernel = np.ones((1, 1), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Merge the building mask into the obstacle mask
    obstacle_mask = cv2.bitwise_and(
        cv2.bitwise_not(final_mask),
        circle_mask
    )
    # Add detected building structures to obstacle mask
    obstacle_mask = cv2.bitwise_or(obstacle_mask, building_mask)
    
    # Remove central bright region from obstacle mask
    obstacle_mask = cv2.bitwise_and(obstacle_mask, cv2.bitwise_not(central_mask))
    
    # Create edge mask (the black border around the fisheye)
    edge_mask = cv2.bitwise_not(circle_mask)
    
    # Modify final sky mask to include central bright region
    # The sky mask should exclude the building areas first
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(building_mask))
    
    # Now, force the central region to be sky
    final_mask = cv2.bitwise_or(final_mask, central_mask)
    
    # Apply the mask to the original image to isolate the sky
    sky_only = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)
    
    # Apply obstacle mask to original image to show obstacles
    obstacles = cv2.bitwise_and(img_rgb, img_rgb, mask=obstacle_mask)
    
    if save_results and output_dir:
        # Save individual masks
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Save masks to files
        cv2.imwrite(os.path.join(masks_dir, "sky_mask.png"), final_mask)
        cv2.imwrite(os.path.join(masks_dir, "obstacle_mask.png"), obstacle_mask)
        cv2.imwrite(os.path.join(masks_dir, "edge_mask.png"), edge_mask)
        cv2.imwrite(os.path.join(masks_dir, "building_mask.png"), building_mask)
        cv2.imwrite(os.path.join(masks_dir, "sky_only.png"), cv2.cvtColor(sky_only, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(masks_dir, "obstacles.png"), cv2.cvtColor(obstacles, cv2.COLOR_RGB2BGR))
        
        # Create visualizations
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Create side-by-side comparison
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(building_mask, cmap='gray')
        axs[0, 1].set_title('Building Structure Mask')
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(obstacles)
        axs[1, 0].set_title('Obstacles (Including Buildings)')
        axs[1, 0].axis('off')
        
        # Display obstacle outline on original image
        contour_img = img_rgb.copy()
        obstacle_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, obstacle_contours, -1, (255, 0, 0), 2)
        axs[1, 1].imshow(contour_img)
        axs[1, 1].set_title('Obstacle Contours')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, "building_detection.png"))
        plt.close()
    
    return final_mask, obstacle_mask, edge_mask, sky_only

def load_sky_mask(masks_dir):
    """
    Load the pre-computed sky mask from file
    
    Args:
        masks_dir: Directory containing the saved masks
    
    Returns:
        tuple: (sky_mask, obstacle_mask, edge_mask, None)
    """
    sky_mask = cv2.imread(os.path.join(masks_dir, "sky_mask.png"), cv2.IMREAD_GRAYSCALE)
    obstacle_mask = cv2.imread(os.path.join(masks_dir, "obstacle_mask.png"), cv2.IMREAD_GRAYSCALE)
    edge_mask = cv2.imread(os.path.join(masks_dir, "edge_mask.png"), cv2.IMREAD_GRAYSCALE)
    
    if sky_mask is None or obstacle_mask is None or edge_mask is None:
        raise ValueError("Could not load one or more mask files")
    
    return sky_mask, obstacle_mask, edge_mask, None

def main():
    # Source directory containing images to test
    source_dir = "D:\\Image"
    
    # Create output directory for test results
    output_dir = "D:\\Image\\test_result\\sky_detection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sky mask from a single image
    reference_image = "D:\\Image\\20250302_012330.jpg"
    print(f"Creating sky mask from reference image: {reference_image}")
    
    # Process the reference image
    sky_mask, obstacle_mask, edge_mask, sky_only = create_sky_mask(
        reference_image, 
        save_results=True,
        output_dir=output_dir
    )
    
    print("Sky mask created and saved successfully")
    print(f"Results saved in: {output_dir}")
    print(f"- Masks: {os.path.join(output_dir, 'masks')}")
    print(f"- Visualizations: {os.path.join(output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()