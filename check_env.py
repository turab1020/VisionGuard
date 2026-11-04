"""
Task 1.1 — Environment verification script.
Run this after installing requirements.txt to confirm everything works.
Take a screenshot of the output for your report.
"""
import sys
import platform

print("=" * 60)
print("VisionGuard — Environment Check (Task 1.1)")
print("=" * 60)

print(f"\nPython version : {sys.version}")
print(f"Platform       : {platform.platform()}")
print(f"Architecture   : {platform.machine()}")

# TensorFlow
try:
    import tensorflow as tf
    print(f"\nTensorFlow     : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs visible   : {len(gpus)}")
    for g in gpus:
        print(f"  - {g}")
    if not gpus:
        print("  (No GPU detected — CPU-only training will be used)")
except ImportError:
    print("\nTensorFlow     : NOT INSTALLED")

# Keras
try:
    import keras
    print(f"Keras          : {keras.__version__}")
except ImportError:
    print("Keras          : NOT INSTALLED")

# Flask
try:
    import flask
    print(f"Flask          : {flask.__version__}")
except ImportError:
    print("Flask          : NOT INSTALLED")

# MLflow
try:
    import mlflow
    print(f"MLflow         : {mlflow.__version__}")
except ImportError:
    print("MLflow         : NOT INSTALLED")

# DVC
try:
    import dvc
    print(f"DVC            : {dvc.__version__}")
except ImportError:
    print("DVC            : NOT INSTALLED")

# Keras Tuner
try:
    import keras_tuner
    print(f"Keras Tuner    : {keras_tuner.__version__}")
except ImportError:
    print("Keras Tuner    : NOT INSTALLED")

# scikit-learn
try:
    import sklearn
    print(f"scikit-learn   : {sklearn.__version__}")
except ImportError:
    print("scikit-learn   : NOT INSTALLED")

# matplotlib
try:
    import matplotlib
    print(f"matplotlib     : {matplotlib.__version__}")
except ImportError:
    print("matplotlib     : NOT INSTALLED")

# Pillow
try:
    import PIL
    print(f"Pillow         : {PIL.__version__}")
except ImportError:
    print("Pillow         : NOT INSTALLED")

# DagsHub
try:
    import dagshub
    print(f"dagshub        : {dagshub.__version__}")
except ImportError:
    print("dagshub        : NOT INSTALLED")

# numpy / pandas
try:
    import numpy as np
    print(f"numpy          : {np.__version__}")
except ImportError:
    print("numpy          : NOT INSTALLED")

try:
    import pandas as pd
    print(f"pandas         : {pd.__version__}")
except ImportError:
    print("pandas         : NOT INSTALLED")

print("\n" + "=" * 60)
print("Environment check complete.")
print("=" * 60)
