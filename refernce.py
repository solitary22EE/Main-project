import numpy as np
import cv2
import os

# Function to compute Information Entropy (IE)
def compute_ie(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return -np.sum(hist * np.log2(hist))

# Function to compute Average Gradient (AG)
def compute_ag(image):
    image = np.float32(image)
    grad_x = np.diff(image, axis=0)  # Gradient along x (vertical)
    grad_y = np.diff(image, axis=1)  # Gradient along y (horizontal)

    # Find the minimum shape to ensure correct element-wise operations
    min_h = min(grad_x.shape[0], grad_y.shape[0])
    min_w = min(grad_x.shape[1], grad_y.shape[1])

    grad_x = grad_x[:min_h, :min_w]
    grad_y = grad_y[:min_h, :min_w]

    ag = np.sqrt((grad_x ** 2 + grad_y ** 2) / 2)
    return np.mean(ag)

# Function to compute UIQM
def compute_uiqm(image):
    def compute_uicm(image):
        """Compute Underwater Image Colorfulness Measure (UICM)."""
        r, g, b = cv2.split(image.astype(np.float32))
        rg = r - g
        yb = (r + g) / 2 - b
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        return np.sqrt(rg_std ** 2 + yb_std ** 2)

    def compute_uism(image):
        """Compute Underwater Image Sharpness Measure (UISM)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap

    def compute_uiconm(image):
        """Compute Underwater Image Contrast Measure (UIConM)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        return contrast

    # Weights for UIQM computation
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = compute_uicm(image)
    uism = compute_uism(image)
    uiconm = compute_uiconm(image)

    return (c1 * uicm + c2 * uism + c3 * uiconm) / 100

# Function to process images in a folder
def evaluate_image_quality(folder_path):
    ie_scores, ag_scores, uiqm_scores = [], [], []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping unreadable image: {filename}")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ie = compute_ie(gray_image)
            ag = compute_ag(gray_image)
            uiqm = compute_uiqm(image)

            ie_scores.append(ie)
            ag_scores.append(ag)
            uiqm_scores.append(uiqm)

            print(f"🔹 {filename} - IE: {ie:.4f}, AG: {ag:.4f}, UIQM: {uiqm:.4f}")

    # Compute average scores for all images
    if ie_scores:
        print("\n📊 Overall Scores for the Folder 📊")
        print(f"Average IE: {np.mean(ie_scores):.4f}")
        print(f"Average AG: {np.mean(ag_scores):.4f}")
        print(f"Average UIQM: {np.mean(uiqm_scores):.4f}")
    else:
        print("No valid images found in the folder.")

# Example usage
folder_path = r"C:\Users\Admin\Downloads\newflowchart\unsharpoutput"  # Change to your folder path
evaluate_image_quality(folder_path)
