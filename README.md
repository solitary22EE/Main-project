# Underwater Image Enhancement using Frequency and Spatial Domain Fusion

This project focuses on enhancing underwater images by combining frequency and spatial domain techniques. It aims to correct color distortion, improve contrast, and enhance visual clarity using a multi-stage approach involving FFT, Rayleigh distribution, and wavelet-based fusion.

## 🌊 Project Overview

Underwater images often suffer from issues like poor visibility, color casts (especially bluish tones), and low contrast. To overcome these challenges, this project implements:

- **Color Correction using LAB Color Space**
- **Contrast Enhancement using Rayleigh Distribution**
- **Fusion using Wavelet Decomposition and Gradient-based Weighting**

## 🚀 Features

- FFT-based frequency domain enhancement
- Rayleigh-based histogram enhancement
- Wavelet transform-based multi-scale fusion
- Supports batch processing of underwater image datasets (UFO-120)

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- PyWavelets (`pywt`)
- Matplotlib (for visualization)

## 📁 Dataset

We use the **UFO-120 Underwater Image Dataset**, which contains various underwater scenes with different color and lighting conditions.

> Note: The dataset is not included in the repo due to size. Please download it from the official source or request access.

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/underwater-image-enhancement.git
   cd underwater-image-enhancement
