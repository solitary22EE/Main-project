import cv2
import numpy as np
import os

def gain_control(image, gain_factors):
    """
    Apply gain control to each color channel of the image.

    Parameters:
    - image: Input image (numpy array).
    - gain_factors: A tuple of gain factors for (R, G, B) channels.

    Returns:
    - Adjusted image (numpy array).
    """
    # Split the image into its color channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply gain control to each channel
    r_channel = cv2.multiply(r_channel, gain_factors[0])
    g_channel = cv2.multiply(g_channel, gain_factors[1])
    b_channel = cv2.multiply(b_channel, gain_factors[2])

    # Clip the values to be in the valid range [0, 255]
    r_channel = np.clip(r_channel, 0, 255).astype(np.uint8)
    g_channel = np.clip(g_channel, 0, 255).astype(np.uint8)
    b_channel = np.clip(b_channel, 0, 255).astype(np.uint8)

    # Merge the channels back together
    adjusted_image = cv2.merge((b_channel, g_channel, r_channel))

    return adjusted_image

# Define the input and output folders
input_folder = r"D:\Mainproject\Main-project\inputimages"  # Replace with your input folder path
output_folder = r"D:\Mainproject\Main-project\outputimages"  # Replace with your output folder path
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Define gain factors for R, G, B channels
gain_factors = (1.5, 1.0, 1.2)  # Example gain factors

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Filter image files
        input_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        # Read the image
        image = cv2.imread(input_path)

        # Apply gain control
        corrected_image = gain_control(image, gain_factors)

        # Save the corrected image
        cv2.imwrite(output_path, corrected_image)
        print(f"Processed and saved: {output_path}")

print("All images have been processed and saved in the output folder.")

