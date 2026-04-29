import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

class DeepfakePredictor:
    def __init__(self, model_path="models/best_model-v3.pt"):
        self.device = torch.device("cpu")  # Force CPU — avoids device mismatch
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self._warmup()

    def _load_model(self, path):
        try:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.4),
                torch.nn.Linear(model.classifier[1].in_features, 2)
            )
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            print("Model loaded on CPU")
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def _warmup(self):
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            self.model(dummy)
        print("Model warmed up!")

    def predict(self, img: Image.Image):
        try:
            tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                conf, pred = torch.max(probs, dim=0)
            return conf.item(), pred.item(), probs
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5, 0, None

    def predict_with_grad(self, img: Image.Image):
        tensor = self.transform(img).unsqueeze(0)
        tensor.requires_grad_(True)
        return tensor