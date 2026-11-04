"""
Task 1.2 — Dataset Audit Script
Walks the dataset and builds a table with: class, file size, width, height,
colour mode, format, and SHA-256 hash for every image.
Detects unreadable (corrupt) files and exact duplicates.
"""
import os
import hashlib
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

def sha256_of_file(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(data_dir: Path) -> pd.DataFrame:
    """Walk every image in the dataset and collect metadata."""
    records = []
    corrupt_files = []

    # The dataset has structure: data/raw/<ClassName>/<ClassName>/<images>
    for class_folder in sorted(data_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name

        # Handle nested folder (AiArtData/AiArtData, RealArt/RealArt)
        inner = class_folder / class_name
        if inner.is_dir():
            image_dir = inner
        else:
            image_dir = class_folder

        print(f"\nScanning class: {class_name} ({image_dir})")

        files = sorted(image_dir.iterdir())
        for fpath in tqdm(files, desc=f"  {class_name}"):
            if not fpath.is_file():
                continue

            file_size = fpath.stat().st_size
            content_hash = sha256_of_file(str(fpath))

            try:
                with Image.open(fpath) as img:
                    img.verify()  # Verify it's a valid image
                # Re-open after verify (verify can leave file in bad state)
                with Image.open(fpath) as img:
                    width, height = img.size
                    colour_mode = img.mode
                    img_format = img.format
                    records.append({
                        "filename": fpath.name,
                        "class": class_name,
                        "file_size_bytes": file_size,
                        "width": width,
                        "height": height,
                        "colour_mode": colour_mode,
                        "format": img_format,
                        "sha256": content_hash,
                        "path": str(fpath),
                        "corrupt": False,
                    })
            except Exception as e:
                corrupt_files.append((fpath.name, class_name, str(e)))
                records.append({
                    "filename": fpath.name,
                    "class": class_name,
                    "file_size_bytes": file_size,
                    "width": None,
                    "height": None,
                    "colour_mode": None,
                    "format": None,
                    "sha256": content_hash,
                    "path": str(fpath),
                    "corrupt": True,
                })

    df = pd.DataFrame(records)
    return df, corrupt_files


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Find exact duplicates based on SHA-256 hash."""
    valid = df[~df["corrupt"]]
    dup_hashes = valid[valid.duplicated(subset="sha256", keep=False)]
    return dup_hashes.sort_values("sha256")


def print_summary(df: pd.DataFrame, corrupt_files: list, duplicates: pd.DataFrame):
    """Print the summary statistics required for the report."""
    valid = df[~df["corrupt"]]

    print("\n" + "=" * 70)
    print("DATASET AUDIT REPORT")
    print("=" * 70)

    # ── Counts per class ──
    print("\n── Counts per Class ──")
    class_counts = df.groupby("class").agg(
        total=("filename", "count"),
        valid=("corrupt", lambda x: (~x).sum()),
        corrupt=("corrupt", "sum"),
    )
    print(class_counts.to_string())
    print(f"\nTotal images: {len(df)}")

    # ── Image dimensions and file size stats ──
    print("\n── Image Dimensions & File Size (valid images only) ──")
    for cls in sorted(valid["class"].unique()):
        subset = valid[valid["class"] == cls]
        print(f"\n  Class: {cls} ({len(subset)} images)")
        for col, label in [("width", "Width"), ("height", "Height"), ("file_size_bytes", "File size (bytes)")]:
            mn = subset[col].min()
            md = subset[col].median()
            mx = subset[col].max()
            print(f"    {label:20s}  min={mn:>10,.0f}   median={md:>10,.0f}   max={mx:>10,.0f}")

    # ── Colour modes ──
    print("\n── Colour Modes ──")
    mode_counts = valid.groupby(["class", "colour_mode"]).size().unstack(fill_value=0)
    print(mode_counts.to_string())

    # ── Formats ──
    print("\n── Image Formats ──")
    format_counts = valid.groupby(["class", "format"]).size().unstack(fill_value=0)
    print(format_counts.to_string())

    # ── Corrupt files ──
    print(f"\n── Corrupt / Unreadable Files: {len(corrupt_files)} ──")
    if corrupt_files:
        for name, cls, err in corrupt_files:
            print(f"  [{cls}] {name}: {err}")
    else:
        print("  None detected.")

    # ── Duplicates ──
    n_dup_groups = duplicates["sha256"].nunique() if len(duplicates) > 0 else 0
    n_dup_files = len(duplicates)
    print(f"\n── Exact Duplicates: {n_dup_files} files in {n_dup_groups} groups ──")
    if len(duplicates) > 0:
        for h, group in duplicates.groupby("sha256"):
            print(f"  Hash {h[:12]}...  ({len(group)} files)")
            for _, row in group.iterrows():
                print(f"    [{row['class']}] {row['filename']}")

    print("\n" + "=" * 70)


def main():
    print("VisionGuard — Dataset Audit (Task 1.2)")
    print(f"Data directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Run download_dataset.py first.")
        sys.exit(1)

    # Run audit
    df, corrupt_files = audit_dataset(DATA_DIR)

    # Find duplicates
    duplicates = find_duplicates(df)

    # Print summary
    print_summary(df, corrupt_files, duplicates)

    # Save full audit table
    audit_csv = OUTPUT_DIR / "dataset_audit.csv"
    df.to_csv(audit_csv, index=False)
    print(f"\nFull audit table saved to: {audit_csv}")

    # Save summary stats
    summary_csv = OUTPUT_DIR / "audit_summary.csv"
    valid = df[~df["corrupt"]]
    summary_rows = []
    for cls in sorted(valid["class"].unique()):
        s = valid[valid["class"] == cls]
        summary_rows.append({
            "class": cls,
            "count": len(s),
            "width_min": s["width"].min(),
            "width_median": s["width"].median(),
            "width_max": s["width"].max(),
            "height_min": s["height"].min(),
            "height_median": s["height"].median(),
            "height_max": s["height"].max(),
            "filesize_min": s["file_size_bytes"].min(),
            "filesize_median": s["file_size_bytes"].median(),
            "filesize_max": s["file_size_bytes"].max(),
            "corrupt_count": len(df[(df["class"] == cls) & df["corrupt"]]),
            "duplicate_count": len(duplicates[duplicates["class"] == cls]),
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Summary stats saved to: {summary_csv}")


if __name__ == "__main__":
    main()
