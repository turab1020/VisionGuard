"""
Step 2 — Download the Kaggle dataset.

Prerequisites:
  1. Install kaggle CLI:  pip install kaggle
  2. Place your kaggle.json at ~/.kaggle/kaggle.json
     (Download from https://www.kaggle.com/settings → API → Create New Token)
  3. Run:  python download_dataset.py

The script downloads and extracts the dataset into data/raw/
"""
import os
import subprocess
import sys

DATASET = "cashbowman/ai-generated-images-vs-real-images"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f"Downloading dataset: {DATASET}")
    print(f"Target directory  : {TARGET_DIR}")
    print()

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", TARGET_DIR, "--unzip"],
            check=True,
        )
        print("\n✅ Dataset downloaded and extracted successfully!")
        print(f"   Location: {TARGET_DIR}")

        # Show what we got
        for root, dirs, files in os.walk(TARGET_DIR):
            level = root.replace(TARGET_DIR, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:
                subindent = " " * 2 * (level + 1)
                for f in files[:5]:
                    print(f"{subindent}{f}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")

    except FileNotFoundError:
        print("❌ 'kaggle' CLI not found. Install it with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("   Make sure ~/.kaggle/kaggle.json exists and is valid.")
        sys.exit(1)

if __name__ == "__main__":
    main()
