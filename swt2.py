import cv2
import numpy as np
import pywt
import os

# Define paths for input and output folders
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\input_images"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\swtoutput1"

# Define Haar wavelet (db1) for SWT
wavelet = 'haar'
level = 1

# Function to apply SWT directly on the RGB image
def apply_swt_rgb(image, beta=1):
    # Apply SWT on the entire RGB image
    coeffs = pywt.swt2(image, wavelet, level=level, axes=(0, 1))
    LL, (LH, HL, HH) = coeffs[0]

    # Apply Gaussian blur to LL (low-frequency component)
    blurred_LL = cv2.GaussianBlur(LL, (5, 5), 0)

    # Calculate difference image (enhanced details)
    detail_enhanced = LL + beta * (LL - blurred_LL)

    # Reconstruct the image with modified LL and original high-frequency bands
    reconstructed_coeffs = [(detail_enhanced, (LH, HL, HH))]
    enhanced_image = pywt.iswt2(reconstructed_coeffs, wavelet, axes=(0, 1))

    # Clip values to valid range
    enhanced_image = np.clip(enhanced_image, 0, 255)
    return np.uint8(enhanced_image)

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the input image
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not open {filename}, skipping...")
            continue
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply SWT on the RGB image and enhance
        enhanced_image = apply_swt_rgb(image_rgb, beta=1)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image_clahe = np.zeros_like(enhanced_image)

        # Apply CLAHE on each channel
        for i in range(3):
            enhanced_image_clahe[:, :, i] = clahe.apply(enhanced_image[:, :, i])
        
        # Convert back to BGR and save the final enhanced image
        final_image = cv2.cvtColor(enhanced_image_clahe, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_image)
        
        print(f"Enhanced image saved as: {output_path}")

print("✅ Batch processing complete. Check the 'output' folder.")
