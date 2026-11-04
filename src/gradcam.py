"""
Task 3.3 (Bonus) — Grad-CAM Visualisation
Generates class activation heatmaps to show which image regions
drive the best model's predictions.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CLASS_NAMES = ["AI-Generated", "Real"]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image."""
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0  # binary — single output neuron
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on the original image."""
    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]

    # Resize heatmap to image size
    jet_heatmap = tf.image.resize(
        jet_heatmap[np.newaxis, ...], (img.shape[0], img.shape[1])
    ).numpy()[0]

    superimposed = jet_heatmap * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # For functional models with nested base models
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    # Return the wrapper layer name for the base model output
                    return layer.name
    return None


def main():
    print("VisionGuard — Grad-CAM Visualisation")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load best model
    model_path = MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        print("ERROR: best_model.keras not found. Run train.py first.")
        sys.exit(1)

    model = keras.models.load_model(model_path)
    print(f"  Loaded model: {model_path}")
    print(f"  Model name: {model.name}")

    # Find the last conv layer — for transfer models, it's inside the base
    # For MobileNetV2_transfer, the base model is the second layer
    last_conv_name = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            # This is the base model (e.g. MobileNetV2)
            # We need to build a special grad model
            base_model = layer
            for sub in reversed(base_model.layers):
                if 'conv' in sub.name.lower() or isinstance(sub, keras.layers.Conv2D):
                    last_conv_name = sub.name
                    break
            break

    if last_conv_name is None:
        # Fallback: try direct layers
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_name = layer.name
                break

    if last_conv_name is None:
        print("ERROR: Could not find a convolutional layer in the model.")
        sys.exit(1)

    print(f"  Last conv layer: {last_conv_name}")

    # For transfer models, we need to access the conv layer through the base
    # Build a grad model that accesses the base model's internals
    base_model_layer = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_model_layer = layer
            break

    if base_model_layer is not None:
        # Transfer model — access base model's conv output
        conv_output = base_model_layer.get_layer(last_conv_name).output
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[conv_output, model.output]
        )
    else:
        # Custom CNN — direct access
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

    # Load test data
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    print(f"  Test set: {X_test.shape[0]} images")

    # Select sample images (3 correct AI, 3 correct Real, 2 misclassified)
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    correct_ai = np.where((y_test == 0) & (y_pred == 0))[0]
    correct_real = np.where((y_test == 1) & (y_pred == 1))[0]
    misclassified = np.where(y_test != y_pred)[0]

    samples = []
    for idx in correct_ai[:3]:
        samples.append((idx, "AI (correct)"))
    for idx in correct_real[:3]:
        samples.append((idx, "Real (correct)"))
    for idx in misclassified[:2]:
        true_label = CLASS_NAMES[int(y_test[idx])]
        pred_label = CLASS_NAMES[int(y_pred[idx])]
        samples.append((idx, f"True={true_label}, Pred={pred_label}"))

    if len(samples) == 0:
        print("  No samples found, using first 8 images")
        samples = [(i, CLASS_NAMES[int(y_test[i])]) for i in range(min(8, len(y_test)))]

    # Generate Grad-CAM for each sample
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, (idx, label) in enumerate(samples):
        img = X_test[idx]
        img_batch = np.expand_dims(img, 0)

        # Get heatmap
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(img_batch)
            loss = pred[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Overlay
        superimposed = overlay_heatmap(img, heatmap)

        conf = float(y_prob[idx])
        pred_label = CLASS_NAMES[1] if conf >= 0.5 else CLASS_NAMES[0]

        # Plot
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{label}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title(f"Heatmap", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(superimposed)
        axes[row, 2].set_title(f"Pred: {pred_label} ({conf:.2f})", fontsize=9)
        axes[row, 2].axis("off")

    fig.suptitle("Grad-CAM — Best Model Attention Maps", fontsize=14, fontweight="bold")
    fig.tight_layout()
    gradcam_path = FIGURES_DIR / "gradcam_heatmaps.png"
    fig.savefig(gradcam_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grad-CAM visualisation saved: {gradcam_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
