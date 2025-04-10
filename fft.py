import cv2
import numpy as np
import os

def apply_fft_correction(img):
    b, g, r = cv2.split(img)

    def process_channel(channel):
        fshift = np.fft.fft2(channel)
        fshift = np.fft.fftshift(fshift)

        avg_val = np.mean(fshift)
        fshift = fshift - avg_val

        f_ishift = np.fft.ifftshift(fshift)
        img_corrected = np.abs(np.fft.ifft2(f_ishift))

        return img_corrected

    b_corrected = process_channel(b)
    g_corrected = process_channel(g)
    r_corrected = process_channel(r)

    corrected_img = cv2.merge((b_corrected, g_corrected, r_corrected))
    corrected_img = cv2.normalize(corrected_img, None, 0, 255, cv2.NORM_MINMAX)

    return corrected_img.astype(np.uint8)

if __name__ == "__main__":
    input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"
    output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\fftoutput"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)

        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load the image {filename}. Skipping.")
            continue

        corrected_img = apply_fft_correction(img)

        corrected_path = os.path.join(output_folder, filename)
        cv2.imwrite(corrected_path, corrected_img)

        # cv2.imshow("Original Image", img)
        # cv2.imshow("FFT Corrected Image", corrected_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
