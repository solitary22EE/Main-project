import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_correction_frequency_domain(image_path, gain_factor=1.2):
    # Read the input image
    image = cv2.imread(r"D:\Mainproject\Main-project\inputimages\set_o46 (1).jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Split the channels
    r, g, b = cv2.split(image)
    
    # Function to apply frequency domain gain control
    def apply_gain(channel, gain):
        # Perform FFT
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        
        # Apply gain
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        new_magnitude = magnitude * gain
        
        # Combine magnitude and phase
        fshift_new = new_magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_new)
        corrected_channel = np.abs(np.fft.ifft2(f_ishift))
        
        # Normalize to 0-255
        corrected_channel = np.clip(corrected_channel, 0, 255).astype(np.uint8)
        return corrected_channel
    
    # Apply gain control to each channel
    r_corrected = apply_gain(r, gain_factor)
    g_corrected = apply_gain(g, gain_factor)
    b_corrected = apply_gain(b, gain_factor)
    
    # Merge corrected channels
    corrected_image = cv2.merge([r_corrected, g_corrected, b_corrected])
    
    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Corrected Image')
    plt.imshow(corrected_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
image_path = "path_to_your_image.jpg"  # Replace with your image path
color_correction_frequency_domain(image_path, gain_factor=1.2)
