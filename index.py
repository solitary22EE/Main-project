import cv2
import numpy as np
import os

# Paths for input and output folders
input_folder = 'input_images'
output_folder = 'output_images'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function for Gain Control in Frequency Domain
def gain_control_frequency_domain(channel, target_intensity=128, max_gain=2.5):
    # Apply DFT to shift to frequency domain
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate the current mean intensity in the frequency domain
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    current_mean = np.mean(magnitude_spectrum)

    # Calculate gain
    gain = target_intensity / current_mean if current_mean > 0 else 1
    gain = min(gain, max_gain)

    # Apply gain in the frequency domain
    dft_shift[:, :, 0] *= gain
    dft_shift[:, :, 1] *= gain

    # Shift back and apply inverse DFT
    dft_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize to 8-bit range
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

# Process each image in the input folder
for image_file in os.listdir(input_folder):
    # Full path to the input image
    input_path = os.path.join(input_folder, image_file)

    # Load the image in RGB format
    image = cv2.imread(input_path)
    if image is None:
        print(f"Skipping {image_file}: Not a valid image.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

    # Split the image into its Red, Green, and Blue channels
    r_channel, g_channel, b_channel = cv2.split(image)

    # Apply gain control in the frequency domain to each channel
    r_corrected = gain_control_frequency_domain(r_channel, max_gain=1.8)
    g_corrected = gain_control_frequency_domain(g_channel, max_gain=1.5)
    b_corrected = gain_control_frequency_domain(b_channel, max_gain=1.2)

    # Merge the corrected channels back
    corrected_image = cv2.merge((r_corrected, g_corrected, b_corrected))

    # Convert back to BGR for saving
    corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)

    # Save the enhanced image to the output folder
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, corrected_image_bgr)
    print(f"Enhanced image saved to {output_path}")
