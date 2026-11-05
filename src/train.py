"""
Tasks 3.1-3.3 + 4.1-4.2 — Unified Training Script (v2 - fixed)
Trains all models, logs to MLflow on DagsHub, compares results.

Fixes from v1:
- Added class weights to handle 55/45 imbalance
- Reduced initial LR (1e-4 for transfer, 5e-4 for CNN)
- Smaller batch size (16) for better gradient signal
- EarlyStopping monitors val_accuracy with patience=7
- More epochs (30 + 15 fine-tune)
- Uses numpy arrays directly instead of tf.data for reliability
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import build_custom_cnn, build_transfer_model, unfreeze_top_layers
from augmentation import build_augmentation_layer

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = ["AI-Generated", "Real"]
BATCH_SIZE = 16
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 1e-4          # lower LR for transfer learning
FINE_TUNE_LR = 1e-5


def setup_mlflow():
    """Set up MLflow tracking (DagsHub or local)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        import mlflow
        import dagshub

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            dagshub.init(
                repo_owner="turab1020",
                repo_name="VisionGuard",
                mlflow=True,
            )
            print(f"  MLflow tracking: {tracking_uri}")
        else:
            mlflow.set_tracking_uri("file://" + str(PROJECT_ROOT / "mlruns"))
            print("  MLflow tracking: local (mlruns/)")

        mlflow.set_experiment("VisionGuard-v2")
        return mlflow
    except Exception as e:
        print(f"  MLflow setup warning: {e}")
        print("  Falling back to local tracking")
        import mlflow
        mlflow.set_tracking_uri("file://" + str(PROJECT_ROOT / "mlruns"))
        mlflow.set_experiment("VisionGuard-v2")
        return mlflow


def load_data():
    """Load preprocessed data splits."""
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Train class distribution: AI={int(np.sum(y_train == 0))}, Real={int(np.sum(y_train == 1))}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_class_weights(y_train):
    """Compute class weights to handle imbalance."""
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.astype(int))
    cw = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class weights: {cw}")
    return cw


def evaluate_model(model, X_test, y_test, model_name, mlflow):
    """Evaluate and log metrics."""
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_auc_roc": float(roc_auc_score(y_test, y_prob)),
    }

    # Log metrics
    mlflow.log_metrics(metrics)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    report_path = RESULTS_DIR / f"{model_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(str(report_path))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    cm_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    print(f"\n  {model_name} Test Results:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    return metrics


def train_model(model_name, model, X_train, y_train, X_val, y_val,
                X_test, y_test, mlflow, class_weights,
                epochs=INITIAL_EPOCHS, lr=LEARNING_RATE,
                base_model=None):
    """Train a single model with MLflow logging."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name):
        # Log params
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("initial_epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "binary_crossentropy")
        mlflow.log_param("class_weights", str(class_weights))
        mlflow.log_param("train_size", len(y_train))
        mlflow.log_param("val_size", len(y_val))
        mlflow.log_param("test_size", len(y_test))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Callbacks — monitor val_accuracy, more patience
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=7,
                restore_best_weights=True, mode="max"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
            ),
        ]

        # Phase 1: Initial training with class weights
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        # Phase 2: Fine-tune (transfer models only)
        if base_model is not None:
            print(f"\n  Fine-tuning {model_name}...")
            mlflow.log_param("fine_tune_epochs", FINE_TUNE_EPOCHS)
            mlflow.log_param("fine_tune_lr", FINE_TUNE_LR)

            unfreeze_top_layers(base_model, num_layers=30)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            ft_callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=5,
                    restore_best_weights=True, mode="max"
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8
                ),
            ]

            ft_history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=FINE_TUNE_EPOCHS,
                callbacks=ft_callbacks,
                class_weight=class_weights,
                verbose=1,
            )

            # Merge histories
            for key in history.history:
                history.history[key].extend(ft_history.history.get(key, []))

        # Log training curves
        total_epochs = len(history.history["loss"])
        for epoch in range(total_epochs):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Training curves figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(history.history["loss"], label="Train")
        ax1.plot(history.history["val_loss"], label="Validation")
        ax1.set_title(f"{model_name} — Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(history.history["accuracy"], label="Train")
        ax2.plot(history.history["val_accuracy"], label="Validation")
        ax2.set_title(f"{model_name} — Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        fig.tight_layout()
        curves_path = FIGURES_DIR / f"training_curves_{model_name}.png"
        fig.savefig(curves_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(curves_path))

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, model_name, mlflow)

        # Save model
        model_path = MODELS_DIR / f"{model_name}.keras"
        model.save(model_path)
        mlflow.log_artifact(str(model_path))
        print(f"  Model saved: {model_path}")

        return metrics, history


def compare_models(all_results):
    """Generate comparison table and figure."""
    import pandas as pd

    rows = []
    for name, metrics in all_results.items():
        rows.append({"Model": name, **{k: round(v, 4) for k, v in metrics.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc_roc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.18
    for i, (name, metrics) in enumerate(all_results.items()):
        values = [metrics[m] for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — VisionGuard v2", fontweight="bold")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels([m.replace("test_", "").replace("_", " ").title()
                         for m in metrics_to_plot])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nComparison chart saved to {FIGURES_DIR / 'model_comparison.png'}")

    best_name = max(all_results, key=lambda n: all_results[n]["test_f1"])
    print(f"\n  Best model by F1: {best_name} (F1={all_results[best_name]['test_f1']:.4f})")
    return best_name


def main():
    print("VisionGuard — Model Training Pipeline v2 (Fixed)")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSetting up MLflow...")
    mlflow = setup_mlflow()

    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("\nComputing class weights...")
    class_weights = get_class_weights(y_train)

    all_results = {}

    # 1. Custom CNN (higher LR since training from scratch)
    model = build_custom_cnn()
    metrics, _ = train_model(
        "CustomCNN", model, X_train, y_train, X_val, y_val, X_test, y_test,
        mlflow, class_weights, epochs=INITIAL_EPOCHS, lr=5e-4
    )
    all_results["CustomCNN"] = metrics
    tf.keras.backend.clear_session()

    # 2. MobileNetV2
    model, base = build_transfer_model("MobileNetV2")
    metrics, _ = train_model(
        "MobileNetV2", model, X_train, y_train, X_val, y_val, X_test, y_test,
        mlflow, class_weights, epochs=INITIAL_EPOCHS, lr=LEARNING_RATE, base_model=base
    )
    all_results["MobileNetV2"] = metrics
    tf.keras.backend.clear_session()

    # 3. EfficientNetB0
    model, base = build_transfer_model("EfficientNetB0")
    metrics, _ = train_model(
        "EfficientNetB0", model, X_train, y_train, X_val, y_val, X_test, y_test,
        mlflow, class_weights, epochs=INITIAL_EPOCHS, lr=LEARNING_RATE, base_model=base
    )
    all_results["EfficientNetB0"] = metrics
    tf.keras.backend.clear_session()

    # 4. ResNet50
    model, base = build_transfer_model("ResNet50")
    metrics, _ = train_model(
        "ResNet50", model, X_train, y_train, X_val, y_val, X_test, y_test,
        mlflow, class_weights, epochs=INITIAL_EPOCHS, lr=LEARNING_RATE, base_model=base
    )
    all_results["ResNet50"] = metrics
    tf.keras.backend.clear_session()

    # Compare
    best_name = compare_models(all_results)

    import shutil
    best_src = MODELS_DIR / f"{best_name}.keras"
    best_dst = MODELS_DIR / "best_model.keras"
    shutil.copy2(best_src, best_dst)
    print(f"\nBest model copied to: {best_dst}")

    with open(RESULTS_DIR / "training_results.json", "w") as f:
        json.dump({"best_model": best_name, "results": all_results}, f, indent=2)
    print(f"Results saved to: {RESULTS_DIR / 'training_results.json'}")


if __name__ == "__main__":
    main()
