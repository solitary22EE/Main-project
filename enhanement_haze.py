import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to compute dark channel
def calculate_dark_channel(image, window_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

# Function to estimate atmospheric light
def estimate_atmospheric_light(image, dark_channel):
    num_pixels = image.shape[0] * image.shape[1]
    num_brightest = max(int(num_pixels * 0.001), 1)
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[-num_brightest:], dark_channel.shape)
    atmospheric_light = np.mean(image[indices], axis=0)
    return atmospheric_light

# Function to estimate transmission map
def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    normalized_image = image / atmospheric_light
    dark_channel = calculate_dark_channel(normalized_image, window_size)
    transmission = 1 - omega * dark_channel
    return transmission

# Function to refine transmission map using guided filter
def refine_transmission(image, transmission, radius=40, epsilon=1e-3):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    transmission_refined = cv2.ximgproc.guidedFilter(guide=gray_image, src=transmission.astype(np.float32), radius=radius, eps=epsilon)
    return transmission_refined

# Function to recover the haze-free image
def recover_image(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.maximum(transmission, t0)
    transmission = np.expand_dims(transmission, axis=2)  # Expand dimensions to match image shape
    recovered = (image - atmospheric_light) / transmission + atmospheric_light
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return recovered


# Function to enhance image quality using FFT
def enhance_image_fft(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a high-pass filter
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.float32)  # Start with all ones
    r = 30  # Radius for the high-pass filter
    cv2.circle(mask, (ccol, crow), r, (0, 0), thickness=-1)  # Zero out the low frequencies

    # Apply the filter
    fshift = dft_shift * mask

    # Perform inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    enhanced = cv2.idft(f_ishift)
    magnitude = cv2.magnitude(enhanced[:, :, 0], enhanced[:, :, 1])

    # Normalize the result to [0, 255]
    enhanced_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced_image


# Main function
def underwater_image_enhancement(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phase 1: Haze removal
    dark_channel = calculate_dark_channel(image)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel)
    transmission = estimate_transmission(image, atmospheric_light)
    refined_transmission = refine_transmission(image, transmission)
    haze_free_image = recover_image(image, refined_transmission, atmospheric_light)

    # Phase 2: Quality enhancement using FFT
    enhanced_image = enhance_image_fft(haze_free_image)

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(image)
    plt.subplot(1, 3, 2), plt.title("Haze-Free Image"), plt.imshow(haze_free_image)
    plt.subplot(1, 3, 3), plt.title("Enhanced Image"), plt.imshow(enhanced_image, cmap='gray')
    plt.show()

# Run the function
underwater_image_enhancement(r"D:\Mainproject\Main-project\inputimages\set_o46 (1).jpg")
