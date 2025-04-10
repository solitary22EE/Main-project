import cv2
import numpy as np
import os

def contrast_stretching_rgb(image):
    """Apply contrast stretching to each channel of an RGB image."""
    b, g, r = cv2.split(image)
    
    # Apply contrast stretching to each channel
    b_stretched = stretch_channel(b)
    g_stretched = stretch_channel(g)
    r_stretched = stretch_channel(r)

    # Merge stretched channels back to RGB image
    enhanced_rgb = cv2.merge((b_stretched, g_stretched, r_stretched))
    return enhanced_rgb

def stretch_channel(channel):
    """Stretch the contrast of a single channel."""
    I_min, I_max = np.min(channel), np.max(channel)
    
    # Avoid division by zero in case I_max == I_min
    if I_max == I_min:
        return channel
    
    # Apply linear contrast stretching
    stretched_channel = ((channel - I_min) / (I_max - I_min) * 255).astype(np.uint8)
    return stretched_channel

def process_rgb_stretching(input_folder, output_folder):
    """Process all images in the input folder and apply only RGB contrast stretching."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Read input image
            image = cv2.imread(input_path)
            if image is None:
                print(f"⚠️ Error reading {filename}, skipping...")
                continue
            
            # Apply RGB contrast stretching
            enhanced_rgb = contrast_stretching_rgb(image)
            
            # Save the final enhanced image
            cv2.imwrite(output_path, enhanced_rgb)
            print(f"✅ RGB Contrast Stretching Applied: {filename}")

    print("🎯 RGB Contrast Stretching successfully applied to all images!")

# Define input and output folders
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"  # Input folder
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\contrast_output"  # Output folder

# Apply only RGB Contrast Stretching to all images
process_rgb_stretching(input_folder, output_folder)
