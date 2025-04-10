import os
import cv2
import matplotlib.pyplot as plt

# Define input, output, and hr folder paths
input_folder = r"C:\Users\Admin\Downloads\newflowchart\input_images"
output_folder = r"C:\Users\Admin\Downloads\newflowchart\unsharpoutput"
hr_folder = r"C:\Users\Admin\Downloads\newflowchart\hr" 


# Check if folders exist
for folder in [input_folder, output_folder, hr_folder]:
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found. Creating: {folder}")
        os.makedirs(folder, exist_ok=True)

# Get a sorted list of files from all folders
input_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
output_files = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
hr_files = sorted([f for f in os.listdir(hr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Find the minimum number of images available
num_images = min(len(input_files), len(output_files), len(hr_files))

if num_images == 0:
    print("❗ No valid images found in one or more folders.")
    exit()

# Display one set of images at a time (input, output, hr)
for i in range(num_images):
    input_path = os.path.join(input_folder, input_files[i])
    output_path = os.path.join(output_folder, output_files[i])
    hr_path = os.path.join(hr_folder, hr_files[i])

    # Read and convert images to RGB for matplotlib
    input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2RGB)
    hr_img = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

    # Plot images side by side
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(input_img)
    plt.title(f"Input: {input_files[i]}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(output_img)
    plt.title(f"Output: {output_files[i]}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hr_img)
    plt.title(f"HR: {hr_files[i]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Wait for key press before displaying next set
    input("Press Enter to display the next set of images...")
