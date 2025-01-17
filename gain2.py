import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def color_correction_frequency_domain(input_folder, hr_image_folder, gain_factor=1.2):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found at path: {input_folder}")

    # Get all image files from the input folder
    input_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Function to apply frequency domain gain control
    def apply_gain(channel, gain):
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        new_magnitude = magnitude * gain
        fshift_new = new_magnitude * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(fshift_new)
        corrected_channel = np.abs(np.fft.ifft2(f_ishift))
        corrected_channel = np.clip(corrected_channel, 0, 255).astype(np.uint8)
        return corrected_channel

    # Process each image in the input folder
    for input_image_name in input_images:
        input_image_path = os.path.join(input_folder, input_image_name)
        
        # Read the input image
        image = cv2.imread(input_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Split the channels
        r, g, b = cv2.split(image)

        r_corrected = apply_gain(r, gain_factor)
        g_corrected = apply_gain(g, gain_factor)
        b_corrected = apply_gain(b, gain_factor)
        corrected_image = cv2.merge([r_corrected, g_corrected, b_corrected])

        # Construct file path for high-resolution image
        hr_image_path = os.path.join(hr_image_folder, input_image_name)

        # PSNR calculations
        psnr_values = {}
        if os.path.exists(hr_image_path):
            hr_image = cv2.imread(hr_image_path)
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
            hr_image = cv2.resize(hr_image, (corrected_image.shape[1], corrected_image.shape[0]))

            # Calculate PSNR for each image combination
            psnr_values["Input vs HR"] = psnr(hr_image, image, data_range=255)
            psnr_values["Corrected vs HR"] = psnr(hr_image, corrected_image, data_range=255)
            psnr_values["Input vs Corrected"] = psnr(corrected_image, image, data_range=255)
        else:
            print(f"High-resolution image not found for: {input_image_name}")
            psnr_values["Input vs HR"] = "N/A"
            psnr_values["Corrected vs HR"] = "N/A"
            psnr_values["Input vs Corrected"] = "N/A"

        # Print PSNR values to the console
        for comparison, value in psnr_values.items():
            if isinstance(value, float):
                print(f"{input_image_name} - {comparison}: {value:.2f} dB")
            else:
                print(f"{input_image_name} - {comparison}: {value}")

        # Display images
        plt.figure(figsize=(15, 5))

        # Display Original Image
        plt.subplot(1, 3, 1)
        psnr_input_vs_hr = psnr_values.get("Input vs HR", "N/A")
        plt.title(f'Original Image - PSNR: {psnr_input_vs_hr if isinstance(psnr_input_vs_hr, str) else f"{psnr_input_vs_hr:.2f} dB"}')
        plt.imshow(image)
        plt.axis('off')

        # Display Corrected Image
        plt.subplot(1, 3, 2)
        psnr_input_vs_corrected = psnr_values.get("Input vs Corrected", "N/A")
        plt.title(f'Corrected Image - PSNR: {psnr_input_vs_corrected if isinstance(psnr_input_vs_corrected, str) else f"{psnr_input_vs_corrected:.2f} dB"}')
        plt.imshow(corrected_image)
        plt.axis('off')

        # Display High-Resolution Image if available
        if os.path.exists(hr_image_path):
            plt.subplot(1, 3, 3)
            psnr_corrected_vs_hr = psnr_values.get("Corrected vs HR", "N/A")
            plt.title(f'High-Resolution Image - PSNR: {psnr_corrected_vs_hr if isinstance(psnr_corrected_vs_hr, str) else f"{psnr_corrected_vs_hr:.2f} dB"}')
            plt.imshow(hr_image)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# Example usage
input_folder = r"D:\Mainproject\Main-project\inputimages"  # Path to input images folder
hr_image_folder = r"D:\Mainproject\Main-project\hr"  # Path to high-resolution images folder

color_correction_frequency_domain(input_folder, hr_image_folder, gain_factor=1.2)
