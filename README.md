# Underwater Image Enhancement

This repository contains the implementation of an underwater image enhancement project using Python. The project aims to improve the visual quality of underwater images by applying advanced color correction and contrast enhancement techniques.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Underwater images often suffer from poor visibility, color distortion, and low contrast due to the absorption and scattering of light in water. This project implements enhancement techniques to address these issues, leveraging:

- **Frequency domain methods** for color correction.
- **Spatial domain methods** for contrast enhancement.

The solution adapts automatically to images with high blue hues and computes gain values based on the input image.

---

## Features
- Automatic gain control for color correction.
- Contrast enhancement using spatial domain techniques.
- Compatibility with the UIBAC underwater image dataset.
- Modular code structure for easy customization and scalability.

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

---

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/underwater-image-enhancement.git
cd underwater-image-enhancement
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

1. Add the underwater images to the `input_images/` directory.
2. Run the script to process the images:

```bash
python enhance.py
```

3. Enhanced images will be saved in the `output_images/` directory.

---

## Dataset
This project uses the [UIBAC underwater image dataset](https://example-link.com). Ensure that the dataset is downloaded and properly organized in the `dataset/` directory before running the code.

---

## Project Workflow

1. **Input Image**: Load the underwater image from the dataset.
2. **Preprocessing**: Remove noise and prepare the image for enhancement.
3. **Color Correction**: Use frequency domain techniques for adjusting color.
4. **Contrast Enhancement**: Apply spatial domain methods to improve contrast.
5. **Output**: Save the enhanced image.

---

## Results
Sample results will be added after running the algorithm on test images.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your changes.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
