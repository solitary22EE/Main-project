import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def sharpen(wbimage, original, sharpness_factor=1.5, blur_radius=2):
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur(blur_radius))
    smoothedr, smoothedg, smoothedb = smoothed_image.split()
    imager, imageg, imageb = wbimage.split()
    
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    smoothedR = np.array(smoothedr, np.float64)
    smoothedG = np.array(smoothedg, np.float64)
    smoothedB = np.array(smoothedb, np.float64)
    
    x, y = wbimage.size
    
    for i in range(y):
        for j in range(x):
            imageR[i][j] = sharpness_factor * imageR[i][j] - (sharpness_factor - 1) * smoothedR[i][j]
            imageG[i][j] = sharpness_factor * imageG[i][j] - (sharpness_factor - 1) * smoothedG[i][j]
            imageB[i][j] = sharpness_factor * imageB[i][j] - (sharpness_factor - 1) * smoothedB[i][j]
    
    imageR = np.clip(imageR, 0, 255)
    imageG = np.clip(imageG, 0, 255)
    imageB = np.clip(imageB, 0, 255)
    
    sharpenIm = np.zeros((y, x, 3), dtype="uint8")         
    sharpenIm[:, :, 0] = imageR
    sharpenIm[:, :, 1] = imageG
    sharpenIm[:, :, 2] = imageB
    
    return Image.fromarray(sharpenIm)

def process_folder(input_folder, output_folder, sharpness_factor=1.5, blur_radius=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            wb_image = img.convert("RGB")  
            sharpened = sharpen(wb_image, img, sharpness_factor, blur_radius)
            
            output_path = os.path.join(output_folder, filename)
            sharpened.save(output_path)
            print(f"Processed: {filename} -> {output_path}")

input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\fusion\ryleoutput"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\unsharpoutput"
process_folder(input_folder, output_folder, sharpness_factor=1.2, blur_radius=3)
