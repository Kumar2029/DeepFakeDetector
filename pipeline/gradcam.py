import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Explicitly target correct EfficientNet conv layer
        target_layer = self.model.features[-1][0]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor, class_idx=None):
        try:
            self.model.eval()
            output = self.model(tensor)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[0, class_idx].backward()

            # Generate heatmap
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam

        except Exception as e:
            print(f"GradCAM error: {e}")
            return np.zeros((224, 224))

    def overlay(self, original_img: Image.Image, cam: np.ndarray):
        """Overlay heatmap on original image"""
        try:
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam), cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            original = np.array(original_img.resize((224, 224)))
            overlayed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
            return Image.fromarray(overlayed)
        except Exception as e:
            print(f"Overlay error: {e}")
            return original_img