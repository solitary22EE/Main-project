import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# 1. Dataset Loading and Preprocessing
def load_image(path):
    """Loads an image and converts it to RGB."""
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def display_image(image, title="Image"):
    """Displays an image using matplotlib."""
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 2. Color Correction (Frequency Domain)
def correct_color_frequency(image):
    """Applies frequency domain corrections for color balance."""
    # Convert to grayscale for frequency analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Perform FFT
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Filter (e.g., low-pass filter to remove high-frequency noise)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    fshift_filtered = fshift * mask
    f_ishift = ifftshift(fshift_filtered)
    corrected_image = np.abs(ifft2(f_ishift))

    # Normalize and merge with original color
    corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    corrected_color = cv2.merge([corrected_image, image[:, :, 1], image[:, :, 2]])
    return corrected_color

# 3. Contrast Enhancement (Spatial Domain)
def enhance_contrast(image):
    """Enhances contrast using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge channels and convert back to RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return enhanced_image

# 4. Dynamic Gain Adjustment
def dynamic_gain_adjustment(image):
    """Adjusts gain dynamically based on pixel intensity."""
    mean_intensity = np.mean(image)
    gain = 128 / mean_intensity if mean_intensity > 0 else 1.0

    adjusted_image = np.clip(image * gain, 0, 255).astype(np.uint8)
    return adjusted_image

# 5. Evaluation and Visualization
def compare_images(original, processed, titles=("Original", "Processed")):
    """Displays side-by-side comparison of images."""
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(titles[0])
    plt.axis("off")

    # Processed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title(titles[1])
    plt.axis("off")

    plt.show()

# Example Workflow
if __name__ == "__main__":
    # Load and preprocess image
    image_path = "input_images/334_img_.png" 
    original_image = load_image(image_path)
    display_image(original_image, title="Original Image")

    # Apply color correction
    color_corrected_image = correct_color_frequency(original_image)
    display_image(color_corrected_image, title="Color Corrected Image")

    # Enhance contrast
    contrast_enhanced_image = enhance_contrast(color_corrected_image)
    display_image(contrast_enhanced_image, title="Contrast Enhanced Image")

    # Adjust gain dynamically
    final_image = dynamic_gain_adjustment(contrast_enhanced_image)
    display_image(final_image, title="Final Enhanced Image")

    # Compare original and final images
    compare_images(original_image, final_image)
