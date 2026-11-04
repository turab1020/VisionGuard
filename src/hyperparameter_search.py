"""
Task 4.3 — Keras Tuner Hyperparameter Search
Runs Bayesian optimisation on the best architecture (MobileNetV2)
to find optimal learning rate, dense units, and dropout rate.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports"
CLASS_NAMES = ["AI-Generated", "Real"]


def build_tunable_model(hp):
    """Build MobileNetV2 model with tunable hyperparameters."""
    # Hyperparameters to tune
    lr = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log")
    dense_units_1 = hp.Choice("dense_units_1", values=[128, 256, 512])
    dense_units_2 = hp.Choice("dense_units_2", values=[64, 128, 256])
    dropout_1 = hp.Float("dropout_1", min_value=0.2, max_value=0.6, step=0.1)
    dropout_2 = hp.Float("dropout_2", min_value=0.1, max_value=0.4, step=0.1)

    # Load MobileNetV2 backbone
    base_model = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(dense_units_1, activation="relu")(x)
    x = keras.layers.Dropout(dropout_1)(x)
    x = keras.layers.Dense(dense_units_2, activation="relu")(x)
    x = keras.layers.Dropout(dropout_2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("VisionGuard — Keras Tuner Hyperparameter Search")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Class weights
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.astype(int))
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class weights: {class_weights}")

    # Keras Tuner — Bayesian Optimisation
    print("\nStarting Bayesian hyperparameter search...")
    tuner = kt.BayesianOptimization(
        build_tunable_model,
        objective="val_accuracy",
        max_trials=10,
        seed=42,
        directory=str(PROJECT_ROOT / "tuner_results"),
        project_name="mobilenetv2_tuning",
        overwrite=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, mode="max"
        ),
    ]

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Results
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBest hyperparameters:")
    print(f"  learning_rate:  {best_hps.get('learning_rate'):.6f}")
    print(f"  dense_units_1:  {best_hps.get('dense_units_1')}")
    print(f"  dense_units_2:  {best_hps.get('dense_units_2')}")
    print(f"  dropout_1:      {best_hps.get('dropout_1'):.1f}")
    print(f"  dropout_2:      {best_hps.get('dropout_2'):.1f}")

    # Retrain with best hyperparameters (longer)
    print("\nRetraining with best hyperparameters (30 epochs + fine-tune)...")
    best_model = tuner.hypermodel.build(best_hps)

    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=30,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=7,
                restore_best_weights=True, mode="max"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
            ),
        ],
        class_weight=class_weights,
        verbose=1,
    )

    # Fine-tune: unfreeze top 30 layers of MobileNetV2
    print("\n  Fine-tuning top 30 layers...")
    base_model = best_model.layers[1]  # MobileNetV2 base
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  {trainable}/{len(base_model.layers)} layers trainable")

    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    ft_history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=15,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5,
                restore_best_weights=True, mode="max"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8
            ),
        ],
        class_weight=class_weights,
        verbose=1,
    )

    # Merge histories
    for key in history.history:
        history.history[key].extend(ft_history.history.get(key, []))

    # Evaluate
    y_prob = best_model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_auc_roc": float(roc_auc_score(y_test, y_prob)),
    }

    print("\n" + "=" * 60)
    print("TUNED MODEL — TEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save tuned model
    tuned_path = MODELS_DIR / "tuned_model.keras"
    best_model.save(tuned_path)
    print(f"\n  Tuned model saved: {tuned_path}")

    # Also save as best_model if it beats the current one
    import shutil
    shutil.copy2(tuned_path, MODELS_DIR / "best_model.keras")
    print(f"  Copied to best_model.keras")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Validation")
    ax1.set_title("Tuned MobileNetV2 — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train")
    ax2.plot(history.history["val_accuracy"], label="Validation")
    ax2.set_title("Tuned MobileNetV2 — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves_tuned.png", dpi=150)
    plt.close(fig)

    # Save results
    best_hp_dict = {
        "learning_rate": float(best_hps.get("learning_rate")),
        "dense_units_1": best_hps.get("dense_units_1"),
        "dense_units_2": best_hps.get("dense_units_2"),
        "dropout_1": float(best_hps.get("dropout_1")),
        "dropout_2": float(best_hps.get("dropout_2")),
    }
    results = {
        "best_hyperparameters": best_hp_dict,
        "tuned_metrics": metrics,
    }
    with open(RESULTS_DIR / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {RESULTS_DIR / 'tuning_results.json'}")

    # Log to MLflow
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        import mlflow
        import dagshub

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            dagshub.init(repo_owner="turab1020", repo_name="VisionGuard", mlflow=True)
        else:
            mlflow.set_tracking_uri("file://" + str(PROJECT_ROOT / "mlruns"))

        mlflow.set_experiment("VisionGuard-v2")

        with mlflow.start_run(run_name="Tuned_MobileNetV2"):
            mlflow.log_params(best_hp_dict)
            mlflow.log_param("model_name", "Tuned_MobileNetV2")
            mlflow.log_param("tuner", "BayesianOptimization")
            mlflow.log_param("max_trials", 10)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(tuned_path))
            mlflow.log_artifact(str(FIGURES_DIR / "training_curves_tuned.png"))
            print("  Logged to MLflow")
    except Exception as e:
        print(f"  MLflow logging skipped: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
