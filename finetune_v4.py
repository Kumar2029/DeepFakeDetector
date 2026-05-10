# finetune_v4.py
# Fine-tunes v3 model on new fake face data → saves as best_model_v4.pt
#
# Run on Google Colab T4:
#   1. Upload best_model-v3.pt
#   2. Upload the training_fake and training_real folders (zip them first)
#   3. Update paths below if needed
#   4. Run script
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os
import random
from pathlib import Path
 
 
# ===== CONFIG =====
MODEL_PATH    = "models/best_model-v3.pt"
SAVE_PATH     = "models/best_model_v4.pt"
 
# New dataset paths
NEW_FAKE_DIR  = r"C:\Users\kumar\Downloads\real-and-fake-face-detection\real_and_fake_face\training_fake"
NEW_REAL_DIR  = r"C:\Users\kumar\Downloads\real-and-fake-face-detection\real_and_fake_face\training_real"
 
# Existing dataset paths
EXISTING_FAKE_DIR = r"D:\DeepFakeDetector\DeepfakeDetector\data\train\fake"
EXISTING_REAL_DIR = r"D:\DeepFakeDetector\DeepfakeDetector\data\train\real"
 
EPOCHS        = 10       # more epochs since dataset is small
BATCH_SIZE    = 32
LR            = 5e-5     # very low LR — small dataset, avoid overwriting v3 knowledge
MAX_EXISTING  = 5000     # sample from existing data to balance
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
print(f"Device: {DEVICE}")
 
 
# ===== DATASET =====
class FaceDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.samples = []
 
        def load_imgs(folder, label, limit=None):
            imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(imgs)
            if limit:
                imgs = imgs[:limit]
            return [(p, label) for p in imgs]
 
        # New fake faces (all 960 — this is what we're teaching)
        new_fake = load_imgs(NEW_FAKE_DIR, label=1)
        print(f"New fake images: {len(new_fake)}")
 
        # New real faces (all available)
        new_real = load_imgs(NEW_REAL_DIR, label=0)
        print(f"New real images: {len(new_real)}")
 
        # Sample from existing data to prevent forgetting
        existing_fake = load_imgs(EXISTING_FAKE_DIR, label=1, limit=MAX_EXISTING)
        existing_real = load_imgs(EXISTING_REAL_DIR, label=0, limit=MAX_EXISTING)
        print(f"Existing fake (sampled): {len(existing_fake)}")
        print(f"Existing real (sampled): {len(existing_real)}")
 
        self.samples = new_fake + new_real + existing_fake + existing_real
        random.shuffle(self.samples)
        print(f"Total dataset: {len(self.samples)} images")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
 
 
# ===== TRANSFORMS =====
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
 
# ===== LOAD MODEL =====
def load_model(path):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    print(f"Loaded: {path}")
    return model.to(DEVICE)
 
 
# ===== TRAIN =====
def train():
    model = load_model(MODEL_PATH)
 
    # Freeze backbone — only train classifier + last 3 feature blocks
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features[-3:].parameters():
        param.requires_grad = True
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
 
    dataset = FaceDataset(train_transform)
 
    # 90/10 split
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
 
    best_val_acc = 0.0
 
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = out.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
 
        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                preds = out.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
 
        train_acc = train_correct / train_total * 100
        val_acc   = val_correct / val_total * 100
        avg_loss  = train_loss / len(train_loader)
        scheduler.step()
 
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved → {SAVE_PATH} (val_acc={val_acc:.1f}%)")
 
    print(f"\nDone. Best val acc: {best_val_acc:.1f}%")
    print(f"Model saved: {SAVE_PATH}")
 
 
if __name__ == "__main__":
    train()