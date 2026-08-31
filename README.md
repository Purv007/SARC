# SARC: Semantically-Aware Regional Compression

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-FF1493?style=flat&logo=yolo&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=ffmpeg&logoColor=white)

**SARC** is a smart multimedia compression system that uses AI to shrink image and video file sizes without making the important parts look blurry.

Normally, when you compress a photo or video to save space, the entire image loses quality equally. SARC fixes this by using Artificial Intelligence to scan the image first. It automatically finds the "important" things—like faces, objects, and text—and protects them at near-perfect quality. Meanwhile, it aggressively compresses the empty background (like sky or walls) to save massive amounts of data. 

## ✨ Features
- **AI-Powered Vision**: Automatically detects objects, text, and structural edges in real-time.
- **Smart Adaptive Compression**: Spends more data on important objects and less data on the background.
- **Smooth Quality Blending**: Gradually fades the compression quality from the background into the objects so it looks natural.
- **Interactive Dashboard**: Upload files, compare the AI compression side-by-side with standard compression, and view your bandwidth savings!

## 🚀 Getting Started

### 1. Backend Setup (FastAPI & AI Models)
*Note: Ensure you have Python 3.10+ installed.*
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 2. Frontend Setup (React UI)
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to access the SARC Dashboard.

## 📊 How it Works (Simply)
1. **Upload**: You drop an image or video into the app.
2. **AI Scan**: The AI scans it and creates a "Heatmap" of what it thinks humans will look at.
3. **Smart Compression**: It compresses the "cold" areas heavily and protects the "hot" areas.
4. **Result**: You get a much smaller file size, but it visually looks just as good as the original!

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
