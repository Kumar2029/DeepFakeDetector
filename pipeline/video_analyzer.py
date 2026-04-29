import cv2
import numpy as np
from PIL import Image

class VideoAnalyzer:
    def __init__(self, num_frames=10):
        self.num_frames = num_frames

    def extract_frames(self, video_path: str):
        """Extract evenly spaced frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                return []

            # Pick evenly spaced frame indices
            indices = np.linspace(
                0, total_frames - 1,
                min(self.num_frames, total_frames),
                dtype=int
            )

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))

            cap.release()
            return frames

        except Exception as e:
            print(f"Video extraction error: {e}")
            return []

    def aggregate_predictions(self, fake_probs: list):
        """
        Use median instead of mean for robustness.
        One bad frame won't skew result.
        """
        if not fake_probs:
            return 0.5, 1

        # Median is more robust than mean
        median_prob = float(np.median(fake_probs))

        if median_prob > 0.6:
            pred = 1      # Deepfake
            conf = median_prob
        elif median_prob < 0.4:
            pred = 0      # Real
            conf = 1 - median_prob
        else:
            pred = 1      # Uncertain → lean fake for safety
            conf = median_prob

        return conf, pred

    def get_frame_results(self, fake_probs: list):
        """Return per-frame results for visualization"""
        results = []
        for i, prob in enumerate(fake_probs):
            label = "🔴 Fake" if prob > 0.5 else "🟢 Real"
            results.append(f"Frame {i+1}: {label} ({prob*100:.1f}%)")
        return "\n".join(results)