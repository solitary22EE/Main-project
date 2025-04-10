import os
import numpy as np
import pywt
import cv2
from PIL import Image

def wavelet_decomposition(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def wavelet_reconstruction(coeffs, wavelet='haar'):
    return pywt.waverec2(coeffs, wavelet)

def calculate_average_gradient(component):
    dx = cv2.Sobel(component, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(component, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(dx**2 + dy**2)
    return np.mean(gradient)

def weighted_wavelet_fusion(img1, img2, wavelet='haar', level=1):
    fused_channels = []
    
    for c in range(3):  # Process R, G, B channels separately
        img1_channel = img1[:, :, c]
        img2_channel = img2[:, :, c]
        
        coeffs1 = wavelet_decomposition(img1_channel, wavelet, level)
        coeffs2 = wavelet_decomposition(img2_channel, wavelet, level)
        
        LL1, (LH1, HL1, HH1) = coeffs1
        LL2, (LH2, HL2, HH2) = coeffs2
        
        grad_V = calculate_average_gradient(LH1) + calculate_average_gradient(LH2)
        grad_H = calculate_average_gradient(HL1) + calculate_average_gradient(HL2)
        grad_D = calculate_average_gradient(HH1) + calculate_average_gradient(HH2)
        
        weight_V = 1 + (grad_V / (grad_V + grad_H + grad_D))
        weight_H = 1 + (grad_H / (grad_V + grad_H + grad_D))
        weight_D = 1 + (grad_D / (grad_V + grad_H + grad_D))
        
        LL_fused = (LL1 + LL2) / 2
        LH_fused = weight_V * LH1 + (1 - weight_V) * LH2
        HL_fused = weight_H * HL1 + (1 - weight_H) * HL2
        HH_fused = weight_D * HH1 + (1 - weight_D) * HH2
        
        fused_coeffs = (LL_fused, (LH_fused, HL_fused, HH_fused))
        fused_channel = wavelet_reconstruction(fused_coeffs, wavelet)
        
        fused_channel = np.clip(fused_channel, 0, 255).astype(np.uint8)
        fused_channels.append(fused_channel)
    
    # Merge the processed channels back into an RGB image
    fused_image = cv2.merge(fused_channels)
    return fused_image


def process_folder(input_folder1, input_folder2, output_folder, wavelet='haar', level=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    filenames = os.listdir(input_folder1)
    for filename in filenames:
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img1_path = os.path.join(input_folder1, filename)
            img2_path = os.path.join(input_folder2, filename)
            
            if not os.path.exists(img2_path):
                print(f"Skipping {filename}: No matching image in second folder.")
                continue
            
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            fused_image = weighted_wavelet_fusion(img1, img2, wavelet, level)
            
            output_path = os.path.join(output_folder, filename)
            Image.fromarray(fused_image).save(output_path)
            print(f"Processed: {filename} -> {output_path}")

input_folder1 = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\ryleoutput"
input_folder2 = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\fftoutput"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\fusion"
process_folder(input_folder1, input_folder2, output_folder, wavelet='haar', level=1)