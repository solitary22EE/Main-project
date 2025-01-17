import cv2
import numpy as np
import os
from skimage.exposure import match_histograms

# Function to apply FFT-based gain control to a single channel
def apply_gain_control(channel, gain):
    f_shift = np.fft.fftshift(np.fft.fft2(channel))
    magnitude = np.abs(f_shift) * gain
    phase = np.angle(f_shift)
    f_new = magnitude * np.exp(1j * phase)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_new)))
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

# Function to equalize the histogram of a channel
def equalize_channel(channel):
    return cv2.equalizeHist(channel)

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate the average intensity of each color channel
def calculate_channel_intensity(image):
    r_channel, g_channel, b_channel = cv2.split(image)
    r_intensity = np.mean(r_channel)
    g_intensity = np.mean(g_channel)
    b_intensity = np.mean(b_channel)
    return r_intensity, g_intensity, b_intensity

# Function to determine if correction is needed
def needs_correction(r_intensity, g_intensity, b_intensity, threshold=0.1):
    overall_mean = (r_intensity + g_intensity + b_intensity) / 3
    deviations = {
        'R': abs(r_intensity - overall_mean) / overall_mean,
        'G': abs(g_intensity - overall_mean) / overall_mean,
        'B': abs(b_intensity - overall_mean) / overall_mean,
    }
    return {channel: deviation > threshold for channel, deviation in deviations.items()}

# Function to match histograms between source and reference images
def histogram_match(source, reference):
    matched = match_histograms(source, reference, channel_axis=-1)  # Updated to use channel_axis
    return matched.astype(np.uint8)

# Main function to process all images in a folder
def process_folder(input_folder, hr_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the output folder if it doesn't exist
    
    for image_name in image_files:
        try:
            image_path = os.path.join(input_folder, image_name)
            hr_path = os.path.join(hr_folder, image_name)  # Corresponding HR image
            
            # Load the original and HR images
            image = load_and_preprocess_image(image_path)
            hr_image = load_and_preprocess_image(hr_path) if os.path.exists(hr_path) else None
            
            # Calculate channel intensities
            r_intensity, g_intensity, b_intensity = calculate_channel_intensity(image)
            
            # Determine if corrections are needed
            corrections_needed = needs_correction(r_intensity, g_intensity, b_intensity)
            
            # Split the image into R, G, B channels
            r_channel, g_channel, b_channel = cv2.split(image)
            
            # Apply corrections selectively
            if corrections_needed['R']:
                r_channel = apply_gain_control(r_channel, 0.2)  # Reduced red intensity further
                r_channel = equalize_channel(r_channel)
            if corrections_needed['G']:
                g_channel = apply_gain_control(g_channel, 0.3)
                g_channel = equalize_channel(g_channel)
            if corrections_needed['B']:
                b_channel = apply_gain_control(b_channel, 0.4)
                b_channel = equalize_channel(b_channel)
            
            # Merge adjusted channels back into an image
            corrected_image = cv2.merge((r_channel, g_channel, b_channel))
            
            # Match histograms if HR image is available
            if hr_image is not None:
                matched_image = histogram_match(corrected_image, hr_image)
            else:
                matched_image = corrected_image
            
            # Save the processed images
            corrected_image_path = os.path.join(output_folder, f"corrected_{image_name}")
            matched_image_path = os.path.join(output_folder, f"matched_{image_name}")
            
            cv2.imwrite(corrected_image_path, cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(matched_image_path, cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR))
            
            print(f"Processed and saved: {image_name}")
        except FileNotFoundError as e:
            print(e)

# Paths to the input, HR, and output folders
input_folder = r"D:\Mainproject\Main-project\inputimages"
hr_folder = r"D:\Mainproject\Main-project\inputimages"
output_folder = r"D:\Mainproject\Main-project\outputimages"

# Run the main function
process_folder(input_folder, hr_folder, output_folder)
