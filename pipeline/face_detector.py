from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np

class FaceDetector:
    def __init__(self, device='cpu'):
        self.mtcnn = MTCNN(
            image_size=224,
            keep_all=False,
            device=device,
            post_process=False,
            select_largest=True
        )

    def extract_face(self, img: Image.Image):
        try:
            face = self.mtcnn(img)
            if face is None:
                return img, False

            # Fix: correct tensor to PIL conversion
            face = face.permute(1, 2, 0).cpu().numpy()
            face = ((face - face.min()) / 
                   (face.max() - face.min()) * 255)
            face = np.clip(face, 0, 255).astype(np.uint8)
            face_img = Image.fromarray(face)
            return face_img, True

        except Exception as e:
            print(f"Face detection error: {e}")
            return img, False