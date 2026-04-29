import os
import shutil
import random

DATASET_PATH = r"C:\Users\kumar\Downloads\real-vs-fake"
PROJECT_DATA = r"D:\DeepFakeDetector\DeepfakeDetector\data"

COUNTS = {
    "train": 60000,
    "validation": 10000
}

DATASET_SPLITS = {
    "train": "train",
    "validation": "valid"
}

for split, source_split in DATASET_SPLITS.items():
    for label in ["real", "fake"]:
        src = os.path.join(DATASET_PATH, source_split, label)
        dst = os.path.join(PROJECT_DATA, split, label)
        
        os.makedirs(dst, exist_ok=True)
        
        all_images = os.listdir(src)
        random.shuffle(all_images)
        selected = all_images[:COUNTS[split]]
        
        print(f"Copying {len(selected)} images to {dst}...")
        
        for img in selected:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

print("\n✅ Done! Data is ready for training.")