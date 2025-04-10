import cv2
import numpy as np
import os

def process_image(image_path, output_folder):
    # Load the color image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert BGR to RGB for proper display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge CLAHE-enhanced L channel back with A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    # Convert back to RGB color space
    final_output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the final output image
    filename = os.path.basename(image_path)
    final_output_path = os.path.join(output_folder, filename)
    
    cv2.imwrite(final_output_path, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    
    print(f"Saved: {final_output_path}")

def process_folder(folder_path, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not image_files:
        print("No image files found in the folder.")
        return
    
    # Correct indentation of the for loop
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        process_image(image_path, output_folder)
    
    print(f"All processed images are saved in the folder: {output_folder}")

# Set folder paths (modify as needed)
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\input_images"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\claheoutput"
process_folder(input_folder, output_folder)

# Print the final output folder path
print(f"Final output folder: {output_folder}")
