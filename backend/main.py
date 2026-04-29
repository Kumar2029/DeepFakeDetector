from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
import shutil, os, base64, io
from PIL import Image

from backend.database import get_db, User, Detection, Base, engine
from backend.auth import (verify_password, hash_password,
                          create_token, decode_token, oauth2_scheme)
from pipeline.predictor import DeepfakePredictor
from pipeline.face_detector import FaceDetector
from pipeline.gradcam import GradCAM
from pipeline.video_analyzer import VideoAnalyzer

# === Init ===
app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Load pipeline
print("Loading pipeline...")
predictor = DeepfakePredictor()
face_detector = FaceDetector()
video_analyzer = VideoAnalyzer(num_frames=10)
gradcam = GradCAM(predictor.model)
print("Pipeline ready!")

os.makedirs("uploads", exist_ok=True)

# === Frontend Routes ===
@app.get("/")
def serve_login():
    return FileResponse("frontend/index.html")

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse("frontend/dashboard.html")

# === Auth Routes ===
@app.post("/auth/register")
def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(username=username, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    token = create_token({"sub": username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": username})
    return {"access_token": token, "token_type": "bearer"}

# === Get Current User ===
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    username = decode_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# === Predict Route ===
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.6),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Save uploaded file
    ext = file.filename.split(".")[-1]
    save_path = f"uploads/{current_user.id}_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import mimetypes
    mime, _ = mimetypes.guess_type(save_path)
    result = {}

    try:
        if mime and mime.startswith("image"):
            img = Image.open(save_path).convert("RGB")
            face_img, face_found = face_detector.extract_face(img)
            conf, pred, probs = predictor.predict(face_img)

            # GradCAM
            tensor = predictor.predict_with_grad(face_img)
            cam = gradcam.generate(tensor, class_idx=pred)
            heatmap_img = gradcam.overlay(face_img, cam)

            # Label with threshold
            if conf < threshold:
                label = "UNCERTAIN"
            else:
                label = "REAL" if pred == 0 else "DEEPFAKE"

            # Convert images to base64
            def img_to_b64(img):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()

            result = {
                "prediction": label,
                "confidence": round(conf * 100, 2),
                "face_detected": face_found,
                "face_image": img_to_b64(face_img),
                "heatmap": img_to_b64(heatmap_img),
                "file_type": "image",
                "frame_results": [],
                "why_fake": get_explanation(label, conf, face_found)
            }

        elif mime and mime.startswith("video"):
            frames = video_analyzer.extract_frames(save_path)
            fake_probs = []
            first_face = None
            first_heatmap = None

            for i, frame in enumerate(frames):
                face_img, face_found = face_detector.extract_face(frame)
                conf, pred, probs = predictor.predict(face_img)
                if probs is not None:
                    fake_probs.append(probs[1].item())
                if i == 0:
                    first_face = face_img
                    tensor = predictor.predict_with_grad(face_img)
                    cam = gradcam.generate(tensor, class_idx=pred)
                    first_heatmap = gradcam.overlay(face_img, cam)

            conf, pred = video_analyzer.aggregate_predictions(fake_probs)
            if conf < threshold:
                label = "UNCERTAIN"
            else:
                label = "REAL" if pred == 0 else "DEEPFAKE"

            def img_to_b64(img):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()

            result = {
                "prediction": label,
                "confidence": round(conf * 100, 2),
                "face_detected": True,
                "face_image": img_to_b64(first_face) if first_face else "",
                "heatmap": img_to_b64(first_heatmap) if first_heatmap else "",
                "file_type": "video",
                "frame_results": video_analyzer.get_frame_results(fake_probs),
                "why_fake": get_explanation(label, conf, True)
            }

        # Save to DB
        detection = Detection(
            user_id=current_user.id,
            filename=file.filename,
            file_type=result["file_type"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            face_detected=result["face_detected"]
        )
        db.add(detection)
        db.commit()

    finally:
        os.remove(save_path)

    return JSONResponse(result)

# === Why Fake Explanation ===
def get_explanation(label, conf, face_found):
    if label == "UNCERTAIN":
        return "Confidence is below threshold. The model could not make a reliable determination. Try a higher quality image with a clearly visible face."
    elif label == "DEEPFAKE":
        return f"The model detected manipulation artifacts in the facial region with {conf*100:.1f}% confidence. The Grad-CAM heatmap highlights suspicious areas in red — these regions show inconsistencies typical of AI-generated or manipulated faces, such as unnatural texture blending, edge artifacts, or lighting inconsistencies."
    else:
        return f"No manipulation artifacts detected. The model classified this as authentic with {conf*100:.1f}% confidence. Facial features appear consistent with natural photography."

# === History Route ===
@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    detections = db.query(Detection).filter(
        Detection.user_id == current_user.id
    ).order_by(Detection.timestamp.desc()).limit(50).all()

    return [{
        "id": d.id,
        "filename": d.filename,
        "file_type": d.file_type,
        "prediction": d.prediction,
        "confidence": d.confidence,
        "face_detected": d.face_detected,
        "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for d in detections]

# === Delete History ===
@app.delete("/history/{detection_id}")
def delete_detection(detection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    detection = db.query(Detection).filter(
        Detection.id == detection_id,
        Detection.user_id == current_user.id
    ).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(detection)
    db.commit()
    return {"message": "Deleted"}