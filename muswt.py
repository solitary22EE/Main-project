import cv2
import numpy as np
import pywt
import os

def apply_swt_enhancement(image_path, beta=1):
    # Read the image
    image = cv2.imread(image_path)
    image = image.astype(np.float32) / 255.0  # Normalize

    # Function to apply SWT enhancement to each channel
    def swt_enhance(channel):
        coeffs = pywt.swt2(channel, wavelet='haar', level=1, axes=(0, 1))
        LL, (LH, HL, HH) = coeffs[0]  # Extract first-level subbands

        # Enhance LL, LH, and HL components
        LL_blurred = cv2.GaussianBlur(LL, (5, 5), 1)
        LLF = LL - LL_blurred
        LLE = LL + beta * LLF

        LH_blurred = cv2.GaussianBlur(LH, (5, 5), 1)
        LHF = LH - LH_blurred
        LHE = LH + beta * LHF

        HL_blurred = cv2.GaussianBlur(HL, (5, 5), 1)
        HLF = HL - HL_blurred
        HLE = HL + beta * HLF

        return pywt.iswt2([(LLE, (LHE, HLE, HH))], wavelet='haar', axes=(0, 1))

    # Split channels
    b, g, r = cv2.split(image)

    # Apply SWT enhancement to each channel
    enhanced_b = swt_enhance(b)
    enhanced_g = swt_enhance(g)
    enhanced_r = swt_enhance(r)

    # Merge channels back
    enhanced_rgb = cv2.merge((enhanced_b, enhanced_g, enhanced_r))

    # Clip values and convert back to 8-bit
    enhanced_rgb = np.clip(enhanced_rgb * 255, 0, 255).astype(np.uint8)

    return enhanced_rgb

#  folder paths
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\swtoutput"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        enhanced_image = apply_swt_enhancement(input_path, beta=1)
        cv2.imwrite(output_path, enhanced_image)