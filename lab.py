import cv2
import numpy as np
import os

def apply_white_balance_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Adjust A and B channels by subtracting their mean
    a = np.clip(a - np.mean(a) + 128, 0, 255).astype(np.uint8)
    b = np.clip(b - np.mean(b) + 128, 0, 255).astype(np.uint8)

    lab_corrected = cv2.merge((l, a, b))
    balanced_img = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return balanced_img

if __name__ == "__main__":
    input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\input_images"
    output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\newflowchart\laboutput"

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

        corrected_img = apply_white_balance_lab(img)

        # Save and display
        corrected_path = os.path.join(output_folder, filename)
        cv2.imwrite(corrected_path, corrected_img)

        # cv2.imshow("Original Image", img)
        # cv2.imshow("LAB White Balanced Image", corrected_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
