import cv2
import numpy as np
import os

# Paths
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\input_images"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\r+woutput"
os.makedirs(output_folder, exist_ok=True)

# Contrast stretching for RGB channels
def stretch_channel(channel):
    """Stretch contrast of a single channel using min-max normalization."""
    min_val = np.min(channel)
    max_val = np.max(channel)
    stretched = (channel - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)

# Apply contrast stretching in RGB space
def contrast_stretching_rgb(image):
    """Apply contrast stretching to each channel of an RGB image."""
    b, g, r = cv2.split(image)
    b_stretched = stretch_channel(b)
    g_stretched = stretch_channel(g)
    r_stretched = stretch_channel(r)
    return cv2.merge((b_stretched, g_stretched, r_stretched))

# White patch algorithm for color correction
def white_patch_correction(image):
    """Apply White Patch Color Correction to an image."""
    b, g, r = cv2.split(image)
    
    # Find max intensity in each channel
    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)
    
    # Avoid division by zero
    if max_b == 0 or max_g == 0 or max_r == 0:
        return image
    
    # Scale each channel to have a maximum intensity of 255
    b = np.clip((b / max_b) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((g / max_g) * 255.0, 0, 255).astype(np.uint8)
    r = np.clip((r / max_r) * 255.0, 0, 255).astype(np.uint8)
    
    corrected_image = cv2.merge((b, g, r))
    return corrected_image

# Process images in the input folder
def process_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error loading image: {input_path}")
                continue

            # Step 1: Apply RGB contrast stretching
            rgb_stretched = contrast_stretching_rgb(image)

            # Step 2: Apply White Patch Correction
            corrected_image = white_patch_correction(rgb_stretched)

            # Save the enhanced image
            cv2.imwrite(output_path, corrected_image)
            print(f"Processed: {filename}")

# Run the script
if __name__ == "__main__":
    process_images(input_folder, output_folder)
    print("✅ RGB stretching and white patch color correction applied to all images.")
