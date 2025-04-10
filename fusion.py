import cv2
import numpy as np
import pywt
import os

def wavelet_decomposition(image, level=1, wavelet='haar'):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    return coeffs

def wavelet_reconstruction(coeffs, wavelet='haar'):
    return pywt.waverec2(coeffs, wavelet=wavelet)

def compute_average_gradient(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    return np.mean(grad)

def weighted_wavelet_fusion(image1, image2, wavelet='haar', level=1):
    fused_channels = []
    
    for i in range(3):  # Process R, G, B channels separately
        coeffs1 = wavelet_decomposition(image1[:, :, i], level, wavelet)
        coeffs2 = wavelet_decomposition(image2[:, :, i], level, wavelet)
        
        cA1, (cH1, cV1, cD1) = coeffs1
        cA2, (cH2, cV2, cD2) = coeffs2
        
        cA_fused = (cA1 + cA2) / 2
        
        G_V1, G_H1, G_D1 = compute_average_gradient(cV1), compute_average_gradient(cH1), compute_average_gradient(cD1)
        G_V2, G_H2, G_D2 = compute_average_gradient(cV2), compute_average_gradient(cH2), compute_average_gradient(cD2)
        
        lambda_V1 = 1 + G_V1 / (G_V1 + G_H1 + G_D1 + 1e-6)
        lambda_H1 = 1 + G_H1 / (G_V1 + G_H1 + G_D1 + 1e-6)
        lambda_D1 = 1 + G_D1 / (G_V1 + G_H1 + G_D1 + 1e-6)
        
        lambda_V2 = 1 + G_V2 / (G_V2 + G_H2 + G_D2 + 1e-6)
        lambda_H2 = 1 + G_H2 / (G_V2 + G_H2 + G_D2 + 1e-6)
        lambda_D2 = 1 + G_D2 / (G_V2 + G_H2 + G_D2 + 1e-6)
        
        cH_fused = (lambda_H1 * cH1 + lambda_H2 * cH2) / 2
        cV_fused = (lambda_V1 * cV1 + lambda_V2 * cV2) / 2
        cD_fused = (lambda_D1 * cD1 + lambda_D2 * cD2) / 2
        
        fused_coeffs = (cA_fused, (cH_fused, cV_fused, cD_fused))
        fused_channel = wavelet_reconstruction(fused_coeffs, wavelet)
        
        fused_channels.append(np.clip(fused_channel, 0, 255).astype(np.uint8))
    
    fused_image = cv2.merge(fused_channels)  # Merge channels to form an RGB image
    return fused_image

def enhance_underwater_images(input_folder1, input_folder2, output_base_folder):
    image_paths1 = sorted([f for f in os.listdir(input_folder1) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    image_paths2 = sorted([f for f in os.listdir(input_folder2) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    output_folder = os.path.join(output_base_folder, os.path.basename(input_folder1))
    os.makedirs(output_folder, exist_ok=True)
    
    for img_name in set(image_paths1) & set(image_paths2):
        img1_path = os.path.join(input_folder1, img_name)
        img2_path = os.path.join(input_folder2, img_name)
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)  # Read as color image
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            print(f"Error: Unable to read image {img_name}")
            continue
        
        fused_image = weighted_wavelet_fusion(img1, img2)
        
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, fused_image)
       # print(f"Enhanced image saved at: {output_path}")

def main():
    input_folder1 = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\ryleoutput"
    input_folder2 = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\fftoutput"
    output_base_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\fusion"
    
    enhance_underwater_images(input_folder1, input_folder2, output_base_folder)
    
if __name__ == "__main__":
    main()
