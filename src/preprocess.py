"""
Task 2.1 — Preprocessing Pipeline
Resizes images to 224x224, normalises to [0,1], converts to RGB,
splits into 70/15/15 train/val/test with stratification.
Saves as NumPy arrays.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

IMG_SIZE = (224, 224)
RANDOM_STATE = 42

# Class mapping: 0 = AI-generated, 1 = Real
CLASS_MAP = {"AiArtData": 0, "RealArt": 1}
CLASS_NAMES = ["AI-Generated", "Real"]


def load_and_preprocess_images(data_dir: Path) -> tuple:
    """Load all images, resize to 224x224, convert to RGB, normalise."""
    images = []
    labels = []
    filenames = []
    skipped = 0

    for class_folder in sorted(data_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in CLASS_MAP:
            continue
        label = CLASS_MAP[class_name]

        # Handle nested folder structure
        inner = class_folder / class_name
        image_dir = inner if inner.is_dir() else class_folder

        print(f"\nLoading class: {class_name} (label={label}) from {image_dir}")

        files = sorted([f for f in image_dir.iterdir() if f.is_file()])
        for fpath in tqdm(files, desc=f"  {class_name}"):
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0  # normalise to [0,1]
                images.append(arr)
                labels.append(label)
                filenames.append(fpath.name)
            except Exception as e:
                skipped += 1
                print(f"  Skipped {fpath.name}: {e}")

    X = np.array(images)
    y = np.array(labels)
    print(f"\nLoaded {len(X)} images, skipped {skipped}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: AI={np.sum(y == 0)}, Real={np.sum(y == 1)}")
    return X, y, filenames


def split_dataset(X, y, filenames):
    """Split into 70/15/15 train/val/test with stratification."""
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
        X, y, filenames,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )
    # Second split: 50/50 of temp → 15% val, 15% test
    X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
        X_temp, y_temp, fn_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    return (X_train, y_train, fn_train), (X_val, y_val, fn_val), (X_test, y_test, fn_test)


def print_split_stats(train, val, test):
    """Print dataset split statistics."""
    for name, (X, y, _) in [("Train", train), ("Val", val), ("Test", test)]:
        total = len(y)
        ai = np.sum(y == 0)
        real = np.sum(y == 1)
        print(f"  {name:5s}: {total:4d} images  |  AI: {ai:3d} ({ai/total*100:.1f}%)  |  Real: {real:3d} ({real/total*100:.1f}%)")


def main():
    print("VisionGuard — Preprocessing Pipeline (Task 2.1)")
    print(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Data source: {DATA_RAW}")

    if not DATA_RAW.exists():
        print(f"ERROR: Data directory not found: {DATA_RAW}")
        sys.exit(1)

    # Load and preprocess
    X, y, filenames = load_and_preprocess_images(DATA_RAW)

    # Split
    print("\nSplitting dataset (70/15/15)...")
    train, val, test = split_dataset(X, y, filenames)

    print("\nSplit statistics:")
    print_split_stats(train, val, test)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "val": val, "test": test}
    for split_name, (X_split, y_split, fn_split) in splits.items():
        np.save(DATA_PROCESSED / f"X_{split_name}.npy", X_split)
        np.save(DATA_PROCESSED / f"y_{split_name}.npy", y_split)
        # Save filenames for traceability
        np.save(DATA_PROCESSED / f"filenames_{split_name}.npy", np.array(fn_split))
        print(f"  Saved {split_name}: X={X_split.shape}, y={y_split.shape}")

    print(f"\nAll splits saved to: {DATA_PROCESSED}")
    print(f"Files: {list(DATA_PROCESSED.glob('*.npy'))}")


if __name__ == "__main__":
    main()
