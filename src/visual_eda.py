"""
Task 1.3 — Visual EDA
Generates distribution plots and sample grids for the dataset audit report.
Saves all figures to reports/figures/
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
AUDIT_CSV = DATA_DIR / "dataset_audit.csv"

# ── Setup ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"AiArtData": "#4C72B0", "RealArt": "#DD8452"}


def load_audit():
    if not AUDIT_CSV.exists():
        print(f"ERROR: {AUDIT_CSV} not found. Run dataset_audit.py first.")
        sys.exit(1)
    df = pd.read_csv(AUDIT_CSV)
    return df[~df["corrupt"]]  # valid images only


def plot_class_balance(df):
    """Bar chart of images per class."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["class"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS.get(c, "#999") for c in counts.index],
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution", fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)
    print("  ✓ class_balance.png")


def plot_dimensions_distribution(df):
    """Histograms of image width and height per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, col, label in zip(axes, ["width", "height"], ["Width (px)", "Height (px)"]):
        for cls in sorted(df["class"].unique()):
            subset = df[df["class"] == cls][col]
            ax.hist(subset, bins=40, alpha=0.6, label=cls, color=COLORS.get(cls, "#999"))
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend()
    axes[0].set_title("Width Distribution", fontweight="bold")
    axes[1].set_title("Height Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dimensions_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ dimensions_distribution.png")


def plot_filesize_distribution(df):
    """Histogram of file sizes per class (in KB)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls in sorted(df["class"].unique()):
        sizes_kb = df[df["class"] == cls]["file_size_bytes"] / 1024
        ax.hist(sizes_kb, bins=50, alpha=0.6, label=cls, color=COLORS.get(cls, "#999"))
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    ax.set_title("File Size Distribution", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "filesize_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ filesize_distribution.png")


def plot_colour_modes(df):
    """Stacked bar chart of colour modes per class."""
    ct = pd.crosstab(df["class"], df["colour_mode"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct.plot(kind="bar", stacked=True, ax=ax, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Colour Modes by Class", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "colour_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ colour_modes.png")


def plot_formats(df):
    """Bar chart of image formats per class."""
    ct = pd.crosstab(df["class"], df["format"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Image Formats by Class", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Format")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_formats.png", dpi=150)
    plt.close(fig)
    print("  ✓ image_formats.png")


def plot_aspect_ratios(df):
    """Scatter plot of width vs height coloured by class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in sorted(df["class"].unique()):
        subset = df[df["class"] == cls]
        ax.scatter(subset["width"], subset["height"], alpha=0.4,
                   s=15, label=cls, color=COLORS.get(cls, "#999"))
    ax.plot([0, df["width"].max()], [0, df["width"].max()],
            "k--", alpha=0.3, label="1:1 ratio")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Aspect Ratios (Width vs Height)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "aspect_ratios.png", dpi=150)
    plt.close(fig)
    print("  ✓ aspect_ratios.png")


def plot_sample_grid(df, n_samples=8):
    """Grid of sample images from each class."""
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), n_samples,
                             figsize=(n_samples * 2, len(classes) * 2.2))

    for row, cls in enumerate(classes):
        subset = df[df["class"] == cls].sample(n=min(n_samples, len(df[df["class"] == cls])),
                                                random_state=42)
        for col, (_, img_row) in enumerate(subset.iterrows()):
            ax = axes[row][col] if len(classes) > 1 else axes[col]
            try:
                img = Image.open(img_row["path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            if col == 0:
                ax.set_title(cls, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sample_grid.png")


def main():
    print("VisionGuard — Visual EDA (Task 1.3)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_audit()
    print(f"Loaded {len(df)} valid images from audit.\n")

    print("Generating plots:")
    plot_class_balance(df)
    plot_dimensions_distribution(df)
    plot_filesize_distribution(df)
    plot_colour_modes(df)
    plot_formats(df)
    plot_aspect_ratios(df)
    plot_sample_grid(df)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"Total plots: {len(list(FIGURES_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
