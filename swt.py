import cv2
import numpy as np
import os
import pywt


# ---------------------- Step 1: Red Channel Compensation ----------------------
def compensate_red_channel(image, red_threshold=0.9):
    """Compensate for the red channel to correct color absorption."""
    b, g, r = cv2.split(image)
    
    # Calculate mean intensities of B, G, R channels
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)

    # Check if red needs enhancement
    if mean_r < red_threshold * (mean_b + mean_g) / 2:
        r_compensated = cv2.addWeighted(r, 1.2, g, -0.1, 0)  # Enhance red channel
    else:
        r_compensated = r
    
    corrected_image = cv2.merge((b, g, r_compensated))
    return corrected_image


# ---------------------- Step 2: Adaptive Color Correction ----------------------
def adaptive_color_correction(image):
    """Adjust color correction using mean and variance normalization."""
    b, g, r = cv2.split(image)
    
    # Calculate mean and variance for each channel
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    std_b, std_g, std_r = np.std(b), np.std(g), np.std(r)

    # Apply color correction using normalization
    b_corr = np.clip((b - mean_b) * (128 / (std_b + 1e-6)) + 128, 0, 255).astype(np.uint8)
    g_corr = np.clip((g - mean_g) * (128 / (std_g + 1e-6)) + 128, 0, 255).astype(np.uint8)
    r_corr = np.clip((r - mean_r) * (128 / (std_r + 1e-6)) + 128, 0, 255).astype(np.uint8)

    corrected_image = cv2.merge((b_corr, g_corr, r_corr))
    return corrected_image


# ---------------------- Step 3: Stationary Wavelet Transform (SWT) ----------------------
def apply_swt(image, wavelet='haar', level=1):
    """Apply Stationary Wavelet Transform (SWT) for detail enhancement."""
    b, g, r = cv2.split(image)

    def swt_channel(channel):
        coeffs = pywt.swt2(channel, wavelet, level=level)
        cA, (cH, cV, cD) = coeffs[0]  # Only take first level coefficients
        
        # Enhance low-frequency component
        enhanced_cA = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced_cA.astype(np.uint8)
    
    # Apply SWT on each channel
    b_enhanced = swt_channel(b)
    g_enhanced = swt_channel(g)
    r_enhanced = swt_channel(r)
    
    enhanced_image = cv2.merge((b_enhanced, g_enhanced, r_enhanced))
    return enhanced_image


# ---------------------- Main Pipeline for Folder Processing ----------------------
def process_images(input_folder, output_folder):
    """Process all images in the input folder and save results in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load image
            image = cv2.imread(input_path)
            if image is None:
                print(f"⚠️ Error loading image: {filename}. Skipping...")
                continue

            # Step 1: Apply Red Channel Compensation
            red_corrected = compensate_red_channel(image)

            # Step 2: Apply Adaptive Color Correction
            color_corrected = adaptive_color_correction(red_corrected)

            # Step 3: Apply Stationary Wavelet Transform
            swt_enhanced = apply_swt(color_corrected)

            # Save the processed image
            cv2.imwrite(output_path, swt_enhanced)
            print(f"✅ Processed: {filename}")

    print("\n🎯 All images have been successfully processed and saved to:", output_folder)


# ---------------------- Folder Locations ----------------------
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"  # Input folder path
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\swtoutptu"  # Output folder path

# Run the processing pipeline
process_images(input_folder, output_folder)
