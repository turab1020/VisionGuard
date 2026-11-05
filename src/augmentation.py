"""
Task 2.2 — Data Augmentation
Adds augmentation layer for training data using tf.keras preprocessing layers.
"""
import tensorflow as tf


def build_augmentation_layer():
    """Build a Sequential augmentation layer for training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def create_datasets(X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size=32, augment=True):
    """Create tf.data.Dataset pipelines with optional augmentation."""
    AUTOTUNE = tf.data.AUTOTUNE

    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1024, seed=42)

    if augment:
        aug_layer = build_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Validation dataset (no augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Test dataset (no augmentation)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA = PROJECT_ROOT / "data" / "processed"

    X_train = np.load(DATA / "X_train.npy")
    y_train = np.load(DATA / "y_train.npy")
    X_val = np.load(DATA / "X_val.npy")
    y_val = np.load(DATA / "y_val.npy")
    X_test = np.load(DATA / "X_test.npy")
    y_test = np.load(DATA / "y_test.npy")

    train_ds, val_ds, test_ds = create_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("Augmentation pipeline ready.")
    print(f"  Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Val batches:   {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"  Test batches:  {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Verify shapes
    for x_batch, y_batch in train_ds.take(1):
        print(f"  Train batch shape: X={x_batch.shape}, y={y_batch.shape}")
