import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image


class DeepfakePredictor:
    def __init__(
        self,
        model_path="models/best_model_v4.pt",
        ensemble_path="models/best_model-v3.pt",
        ensemble=False,  # disabled — v4 alone until ensemble weights are tuned
    ):
        self.device = torch.device("cpu")
        self.ensemble = ensemble
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        print("Loading primary model (v4 / diverse fakes)...")
        self.model_primary = self._load_model(model_path)

        if self.ensemble:
            print("Loading ensemble model (v3 / FF++)...")
            self.model_ensemble = self._load_model(ensemble_path)

        self._warmup()

    def _load_model(self, path):
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(model.classifier[1].in_features, 2)
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"  Loaded: {path}")
        return model

    def _warmup(self):
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            self.model_primary(dummy)
            if self.ensemble:
                self.model_ensemble(dummy)
        print("Models warmed up!")

    def _infer(self, model, tensor):
        """Run single model, return probs [real_prob, fake_prob]."""
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
        return probs

    def predict(self, img: Image.Image):
        tensor = self.transform(img).unsqueeze(0)

        probs_primary = self._infer(self.model_primary, tensor)

        if self.ensemble:
            probs_ensemble = self._infer(self.model_ensemble, tensor)

            # Weighted ensemble:
            #   v4 (primary) — diverse fakes, StyleGAN variants → weight 0.75
            #   v3 (ensemble) — FF++ / FaceSwap               → weight 0.25
            fake_score = 0.75 * probs_primary[1].item() + 0.25 * probs_ensemble[1].item()
            real_score = 1.0 - fake_score

            blended = torch.tensor([real_score, fake_score])
            pred = 1 if fake_score > real_score else 0
            conf = max(fake_score, real_score)
            return conf, pred, blended

        else:
            conf, pred = torch.max(probs_primary, dim=0)
            return conf.item(), pred.item(), probs_primary

    def predict_with_grad(self, img: Image.Image):
        """Returns tensor with grad enabled (used by GradCAM on primary model)."""
        tensor = self.transform(img).unsqueeze(0)
        tensor.requires_grad_(True)
        return tensor