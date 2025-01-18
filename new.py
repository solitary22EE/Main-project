import cv2
import numpy as np

def color_correction(input_path, output_path):
    # Load the input image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load input image from {input_path}")

    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge the channels back
    lab_corrected = cv2.merge((l_clahe, a, b))

    # Convert back to BGR color space
    corrected_image = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # Save the corrected image
    cv2.imwrite(output_path, corrected_image)

def image_enhancement(input_path, output_path):
    # Load the color-corrected image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load input image from {input_path}")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert to HSV color space for better contrast handling
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Enhance the V (brightness) channel
    h, s, v = cv2.split(hsv)
    v_enhanced = cv2.equalizeHist(v)
    hsv_enhanced = cv2.merge((h, s, v_enhanced))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # Apply unsharp masking for sharpening
    gaussian_blur = cv2.GaussianBlur(enhanced_image, (9, 9), 10.0)
    sharpened = cv2.addWeighted(enhanced_image, 1.5, gaussian_blur, -0.5, 0)

    # Save the enhanced image
    cv2.imwrite(output_path, sharpened)

# Paths
input_image_path = r'D:\Mainproject\Main-project\inputimages\set_f17.jpg'  # Input image
color_corrected_path = r'D:\Mainproject\Main-project\outputimages\color_corrected.jpg'  # Output color-corrected image
enhanced_image_path = r'D:\Mainproject\Main-project\outputimages\enhanced_image.jpg'  # Output fully enhanced image

# Perform color correction
color_correction(input_image_path, color_corrected_path)
print("Color correction completed. Color-corrected image saved.")

# Perform image enhancement
image_enhancement(color_corrected_path, enhanced_image_path)
print("Image enhancement completed. Fully enhanced image saved.")
