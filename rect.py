import os
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

# Paths
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\input_images"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\rectout"
os.makedirs(output_folder, exist_ok=True)

# Parameters (set these based on experiments)
beta = 0.1
lambda_ = 10
alpha = 100
gamma = 1

# Horizontal and vertical difference operators
Dx = np.array([[1, -1]])
Dy = np.array([[1], [-1]])

# FFT helper functions
def fft(img):
    return fftshift(fft2(img))

def ifft(img):
    return np.real(ifft2(fftshift(img)))

def gaussian_low_pass(img, kernel_size=15):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def retinex_process(image, beta, lambda_, alpha, gamma):
    # Convert to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    Lccr = lab[:, :, 0].astype(np.float32)

    # Initialization of Idccr and Rccr
    Idccr = gaussian_low_pass(Lccr)
    Rccr = np.zeros_like(Lccr)

    Io = np.mean(Idccr)

    # Resize Dx and Dy to match the shape of Lccr
    Dx_padded = np.zeros_like(Lccr)
    Dy_padded = np.zeros_like(Lccr)
    Dx_padded[:Dx.shape[0], :Dx.shape[1]] = Dx
    Dy_padded[:Dy.shape[0], :Dy.shape[1]] = Dy

    # Precompute FFTs of Dx and Dy
    F_Dx = fft(Dx_padded)
    F_Dy = fft(Dy_padded)
    F1 = np.ones(Lccr.shape)

    for _ in range(10):  # Iterations for optimization
        # Update Rccr
        numerator_R = fft(Lccr / Idccr)
        denominator_R = F1 + beta * lambda_ * (np.conj(F_Dx) * F_Dx + np.conj(F_Dy) * F_Dy)
        Rccr = ifft(numerator_R / denominator_R)

        # Update Idccr
        numerator_I = fft(gamma * Io + Lccr / Rccr)
        denominator_I = F1 + alpha * (np.conj(F_Dx) * F_Dx + np.conj(F_Dy) * F_Dy)
        Idccr = ifft(numerator_I / denominator_I)

    # Compute the restored image
    Ire = Rccr * Idccr

    # Rescale to 8-bit
    Ire_rescaled = np.clip(Ire, 0, 255).astype(np.uint8)
    lab[:, :, 0] = Ire_rescaled

    # Convert back to BGR
    restored_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return restored_image

# Process each image
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to read {input_path}")
        continue

    # Apply Retinex process
    restored_image = retinex_process(image, beta, lambda_, alpha, gamma)

    # Save the output image
    cv2.imwrite(output_path, restored_image)
    print(f"Processed and saved: {output_path}")
