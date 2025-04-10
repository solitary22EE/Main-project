import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Corrected folder paths
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\ryleoutput"

# Rayleigh distribution parameter and intensity range
alpha = 0.4  # Rayleigh distribution parameter
O_min, O_max = 0, 255  # Full 8-bit range

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def rayleigh_distribution_enhancement(image, alpha=0.4):
    # Convert to YCrCb color space to work on luminance
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Convert to float for calculations
    Y = Y.astype(np.float32)

    # Compute global dynamic range
    I_min = np.min(Y)
    I_max = np.max(Y)
    I_c = I_max - I_min

    # Define intensity thresholds based on 3.5% and 96.5% rules
    min_threshold = 0.035 * I_c
    max_threshold = 0.965 * I_c

    # Clip minimum and maximum intensities
    min_intensity = max(I_min, min_threshold)
    max_intensity = min(I_max, max_threshold)

    # Normalize the image to range [0, 1]
    normalized_Y = (Y - I_min) / (I_max - I_min + 1e-6)

    # Apply Rayleigh distribution mapping
    PDFR = (normalized_Y / (alpha * 2)) * np.exp(-normalized_Y**2 / (2 * alpha**2))
    CDFR = 1 - np.exp(-normalized_Y**2 / (2 * alpha**2))

    # Use the CDF for remapping intensities
    stretched_Y = O_min + CDFR * (O_max - O_min)

    # Clip to valid range
    stretched_Y = np.clip(stretched_Y, O_min, O_max).astype(np.uint8)

    # Merge with Cr and Cb channels and convert back to BGR
    enhanced_ycrcb = cv2.merge([stretched_Y, Cr, Cb])
    enhanced_image = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)

    return enhanced_image

# Process images in the input folder
sorted_filenames = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))])

# Process each sorted image
for filename in sorted_filenames:
    # Read the image
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❗ Error loading image: {image_path}")
        continue

    # Perform Rayleigh distribution enhancement
    enhanced_image = rayleigh_distribution_enhancement(image)

    # Save the enhanced image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, enhanced_image)

print(f"✅ Rayleigh distribution enhancement completed. Enhanced images are saved in: {output_folder}")
