# SARC: Semantically-Aware Regional Compression

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white)

SARC is a revolutionary multimedia compression system that abandons traditional uniform video and image quantization in favor of **AI-driven, spatially-varying semantic compression**. 

By utilizing an ensemble of specialized neural networks (YOLOv8, EasyOCR), SARC dynamically computes a continuous 2D "Importance Heatmap" for any image or video frame. This ensures that visually and semantically critical regions (e.g., faces, text, moving objects) are preserved at near-lossless quality, while irrelevant background data is aggressively compressed, saving bandwidth without destroying perceptual quality.

## ✨ Features
- **Multi-Modal AI Engine**: Identifies physical entities, high-frequency textual data, and spatial variances in real-time.
- **Continuous Importance Mapping**: Converts bounding boxes into Gaussian decay gradients for smooth, non-jarring compression transitions.
- **Adaptive Spatially-Varying Quantization**: Mathematically maps importance scores `[0.0 to 1.0]` directly to localized codec quality parameters (e.g., JPEG Q-factors).
- **Perceptual Metrics Dashboard**: Evaluates success using human-centric perceptual metrics (LPIPS, SSIM) rather than purely mathematical distortion (PSNR).
- **Full History & Diagnostics**: SQLite-backed history panel tracking compression ratios, bandwidth savings, and exact AI detections for every uploaded asset.

## 🚀 Getting Started

### 1. Backend Setup (FastAPI & AI Models)
*Note: Ensure you have Python 3.10+ and a CUDA-capable GPU for faster processing.*
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 2. Frontend Setup (React & Vite)
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to access the SARC Dashboard.

## 📊 How it Works
1. **Upload**: Drag and drop an image or MP4 video.
2. **Analysis**: The SARC engine extracts frames and passes them through YOLOv8 and EasyOCR.
3. **Heatmap Generation**: A 2D spatial heatmap is generated representing the "importance" of every pixel.
4. **Adaptive Compression**: The background is heavily compressed, while the important semantic regions are preserved.
5. **Compare**: Use the interactive slider to compare SARC vs. Uniform compression, and view real-time LPIPS/SSIM metrics.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
