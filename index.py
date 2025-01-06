import cv2
import numpy as np
import pywt

def wavelet_transform_enhancement(image_path):
    # Load the underwater image
    img = cv2.imread("input_images\\8_img_.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to YCrCb color space
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Apply DWT to the luminance channel
    coeffs2 = pywt.dwt2(y, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Enhance the high-frequency components
    LH_enhanced = cv2.equalizeHist(np.uint8(LH))
    HL_enhanced = cv2.equalizeHist(np.uint8(HL))
    HH_enhanced = cv2.equalizeHist(np.uint8(HH))

    # Reconstruct the image using the enhanced components
    coeffs2_enhanced = (LL, (LH_enhanced, HL_enhanced, HH_enhanced))
    y_enhanced = pywt.idwt2(coeffs2_enhanced, 'haar')

    # Merge the enhanced luminance channel back with Cr and Cb channels
    img_ycrcb_enhanced = cv2.merge((np.uint8(y_enhanced), cr, cb))
    img_enhanced = cv2.cvtColor(img_ycrcb_enhanced, cv2.COLOR_YCrCb2RGB)

    return img_enhanced

# Example usage
enhanced_image = wavelet_transform_enhancement('underwater_image.jpg')
cv2.imwrite('enhanced_image.jpg', cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
