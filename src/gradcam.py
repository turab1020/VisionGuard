"""
Task 3.3 (Bonus) -- Grad-CAM Visualisation
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


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on the original image."""
    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = tf.image.resize(
        jet_heatmap[np.newaxis, ...], (img.shape[0], img.shape[1])
    ).numpy()[0]
    superimposed = jet_heatmap * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


def compute_gradcam(model, img_batch, conv_layer_name):
    """
    Compute Grad-CAM by rebuilding a functional model that traces
    from the top-level input through the base model's conv layer.
    Works for nested transfer-learning models.
    """
    # Find if there's a nested base model
    base_layer = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_layer = layer
            break

    if base_layer is not None:
        # Transfer model: rebuild the forward pass as a single graph
        inp = keras.Input(shape=model.input_shape[1:])

        # Build a sub-model of the base that also outputs the conv layer
        inner_conv_out = base_layer.get_layer(conv_layer_name).output
        base_with_conv = keras.Model(
            inputs=base_layer.input,
            outputs=[inner_conv_out, base_layer.output],
            name="base_with_conv"
        )

        # Trace through the top-level model
        x = inp
        conv_out = None
        for layer in model.layers[1:]:  # skip InputLayer
            if layer.name == base_layer.name:
                conv_out, x = base_with_conv(x)
            else:
                x = layer(x)

        grad_model = keras.Model(inputs=inp, outputs=[conv_out, x])
    else:
        # Plain CNN: direct access
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layer_name).output, model.output]
        )

    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(img_batch, training=False)
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap, float(pred[0, 0])


def main():
    print("VisionGuard -- Grad-CAM Visualisation")
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

    # Find last conv layer name
    conv_layer_name = None
    base_name = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_name = layer.name
            for sub in layer.layers:
                if isinstance(sub, keras.layers.Conv2D):
                    conv_layer_name = sub.name
            break

    if conv_layer_name is None:
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                conv_layer_name = layer.name

    if conv_layer_name is None:
        print("ERROR: No convolutional layer found.")
        sys.exit(1)

    if base_name:
        print(f"  Base model: {base_name}")
    print(f"  Last conv layer: {conv_layer_name}")

    # Load test data
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    print(f"  Test set: {X_test.shape[0]} images")

    # Get predictions
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Select samples: 3 correct AI, 3 correct Real, 2 misclassified
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
        samples = [(i, CLASS_NAMES[int(y_test[i])]) for i in range(min(8, len(y_test)))]

    print(f"  Generating Grad-CAM for {len(samples)} samples...")

    # Generate Grad-CAM
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, (idx, label) in enumerate(samples):
        img = X_test[idx]
        img_batch = np.expand_dims(img, 0)

        heatmap, prob = compute_gradcam(model, img_batch, conv_layer_name)
        superimposed = overlay_heatmap(img, heatmap)

        pred_label = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{label}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title("Heatmap", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(superimposed)
        axes[row, 2].set_title(f"Pred: {pred_label} ({prob:.2f})", fontsize=9)
        axes[row, 2].axis("off")

    fig.suptitle("Grad-CAM -- Best Model Attention Maps", fontsize=14, fontweight="bold")
    fig.tight_layout()
    gradcam_path = FIGURES_DIR / "gradcam_heatmaps.png"
    fig.savefig(gradcam_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grad-CAM visualisation saved: {gradcam_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
