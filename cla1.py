import cv2
import os
import numpy as np


def get_rgb2lab(R, G, B):
    # Convert RGB to LAB using OpenCV
    img = cv2.merge([R, G, B])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img)
    return L.astype(np.float64), a.astype(np.float64), b.astype(np.float64)


def get_color_cast(im):
    R, G, B = cv2.split(im)
    L, a, b = get_rgb2lab(R, G, B)
   
    var_ansa = np.var(a)
    var_ansb = np.var(b)
   
    var_sq = np.sqrt(var_ansa + var_ansb)
    u = np.sqrt(np.mean(a) ** 2 + np.mean(b) ** 2)
    D = u - var_sq
    Dl = D / var_sq if var_sq != 0 else 0
   
    return D, Dl


def CC(I):
    h, w, _ = I.shape
    imsize = h * w
    Ivec = I.reshape(imsize, 3)
    avgIC = np.mean(Ivec, axis=0)
    _, Dl = get_color_cast(I)
   
    if Dl <= 0:
        sc = np.array([1, 1, 1])
    else:
        sc = (np.maximum(np.max(avgIC), 0.1) / np.maximum(avgIC, 0.1)) ** (1 / max(np.sqrt(Dl), 1))
   
    return sc


def apply_color_correction(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
   
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
           
            scale_factors = CC(img)
            corrected_img = np.clip(img * scale_factors, 0, 255).astype(np.uint8)
            corrected_img = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
           
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, corrected_img)
            print(f"Processed: {filename}")


# Example usage:
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"
output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\myout"
apply_color_correction(input_folder, output_folder)