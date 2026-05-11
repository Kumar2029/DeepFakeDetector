# DeepFake Detector

An AI-powered deepfake detection system using EfficientNet-B0, MTCNN face detection, Grad-CAM heatmaps, and a FastAPI backend with JWT authentication.

---

## Features

- Image deepfake detection (JPG, PNG)
- Video deepfake detection (MP4, MOV) — 10-frame analysis
- Grad-CAM heatmap visualization
- MTCNN face detection
- MediaPipe face mesh overlay
- Webcam live capture + analysis
- JWT authentication (register/login)
- Detection history with stats
- Ensemble model (v4 primary + v3 secondary)

---

## Requirements

- Python 3.11
- CUDA-compatible GPU (optional but recommended)
- Git
- FFmpeg (required for video analysis)

---

## Project Structure

```
DeepFakeDetector/
├── backend/
│   ├── main.py              ← FastAPI app, routes
│   ├── auth.py              ← JWT auth
│   └── database.py          ← SQLAlchemy models
├── pipeline/
│   ├── predictor.py         ← EfficientNet ensemble
│   ├── face_detector.py     ← MTCNN face detection
│   ├── gradcam.py           ← Grad-CAM heatmap
│   └── video_analyzer.py    ← Frame extraction + aggregation
├── frontend/
│   ├── index.html           ← Login page
│   ├── dashboard.html       ← Main UI
│   └── static/
│       └── mediapipe/       ← Self-hosted MediaPipe files
├── models/
│   ├── best_model_v4.pt     ← Primary model (diverse fakes) ✅ included
│   ├── best_model-v3.pt     ← Ensemble model (FF++) ✅ included
│   └── backup/
│       └── best_model_v2.pt ← Backup model ✅ included
└── uploads/                 ← Temp upload directory (auto-created)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DeepFakeDetector.git
cd DeepFakeDetector
```

### 2. Install FFmpeg

FFmpeg is required for video frame extraction.

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract and add `bin/` folder to system PATH
3. Verify: `ffmpeg -version`

**Linux:**
```bash
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### 3. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

**With GPU (CUDA 12.1):**
```bash
pip install fastapi uvicorn python-multipart sqlalchemy
pip install passlib[bcrypt] python-jose[cryptography]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install facenet-pytorch opencv-python pillow numpy requests
```

**CPU only (no GPU):**
```bash
pip install fastapi uvicorn python-multipart sqlalchemy
pip install passlib[bcrypt] python-jose[cryptography]
pip install torch torchvision
pip install facenet-pytorch opencv-python pillow numpy requests
```

> Note: First run downloads EfficientNet ImageNet weights (~20MB) automatically.

### 5. Run the Server

```bash
py -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Windows PowerShell (if execution policy error):**
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\Activate.ps1
py -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
Loading pipeline...
Loading primary model (v4 / diverse fakes)...
  Loaded: models/best_model_v4.pt
Loading ensemble model (v3 / FF++)...
  Loaded: models/best_model-v3.pt
Models warmed up!
Pipeline ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 6. Open the App

Navigate to: [http://localhost:8000](http://localhost:8000)

Register a new account → log in → start detecting.

> Note: The SQLite database (`deepfake.db`) is created automatically on first run.

---

## Usage

### Image Analysis
1. Click **Upload** tab
2. Drag and drop or click to select a JPG/PNG file
3. Adjust confidence threshold (default 60%)
4. Click **Analyze file**
5. View verdict, confidence score, face crop, and Grad-CAM heatmap

### Video Analysis
1. Upload an MP4 or MOV file
2. System extracts 10 frames, runs model on each, aggregates results
3. Frame-by-frame scores shown below the result
4. Note: 20MB video takes ~30-60 seconds on CPU

### Webcam
1. Click **Webcam** tab
2. Click camera icon to start
3. Click **Capture & Analyze** to analyze current frame

---

## API Reference

All endpoints except `/auth/register` and `/auth/login` require Bearer token.

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login, returns JWT token |

### Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Analyze image or video |
| GET | `/history` | Get detection history |
| DELETE | `/history/{id}` | Delete a record |

### `/predict` Request

```
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <image or video file>
threshold: 0.45  (optional, default 0.45)
```

### `/predict` Response

```json
{
  "prediction": "DEEPFAKE",
  "confidence": 87.3,
  "face_detected": true,
  "face_image": "<base64 PNG>",
  "heatmap": "<base64 PNG>",
  "file_type": "image",
  "frame_results": [],
  "why_fake": "The model detected manipulation artifacts..."
}
```

---

## Troubleshooting

**Server won't start**
- Ensure venv is activated
- Check Python version: `python --version` (must be 3.11)
- Verify all model files exist in `models/`

**Video analysis not working**
- Install FFmpeg and add to PATH (see Step 2)
- Verify: `ffmpeg -version` in terminal

**"Module not found" error**
- Activate venv first: `venv\Scripts\activate`
- Reinstall missing package: `pip install <package>`

**Port 8000 already in use**
```bash
# Windows — find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**401 Unauthorized**
- Token expired — log out and log back in
- Clear browser storage: open console (F12) → `localStorage.clear()` → refresh

**Face mesh not showing**
- Check browser console (F12) for CDN errors
- Set up self-hosted MediaPipe in `frontend/static/mediapipe/`

**Fake images showing as REAL**
- Lower confidence threshold to 40% using the UI slider
- Model covers: StyleGAN1/2, FaceSwap (FF++), diverse GAN fakes
- Model may miss: StyleGAN3, very high quality commercial deepfakes

---

## Model Details

| Model | Training Data | Best Accuracy | Covers |
|-------|--------------|---------------|--------|
| v4 (primary) | 140k Kaggle + 960 diverse fakes | 84.6% | StyleGAN, diverse GANs |
| v3 (ensemble) | 220k Kaggle + FF++ | 98.9% | FaceSwap, FaceForensics++ |

**Ensemble formula:**
```
fake_score = 0.75 × v4_fake_prob + 0.25 × v3_fake_prob
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| ML Framework | PyTorch |
| Model | EfficientNet-B0 |
| Face Detection | MTCNN (facenet-pytorch) |
| Heatmap | Grad-CAM |
| Face Mesh | MediaPipe |
| Database | SQLite + SQLAlchemy |
| Auth | JWT (python-jose) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — video deepfake dataset
- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) — GAN face dataset
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation
- [MediaPipe](https://mediapipe.dev) — face mesh