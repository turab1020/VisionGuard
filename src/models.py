"""
Task 3.1 & 3.2 — Model Definitions
Custom CNN baseline + Transfer Learning models (MobileNetV2, EfficientNetB0, ResNet50)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_custom_cnn(input_shape=(224, 224, 3)):
    """Task 3.1 — Custom CNN from scratch."""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="custom_cnn")
    return model


def build_transfer_model(backbone_name, input_shape=(224, 224, 3), fine_tune_layers=20):
    """Task 3.2 — Transfer learning with pretrained backbone."""
    backbones = {
        "MobileNetV2": keras.applications.MobileNetV2,
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "ResNet50": keras.applications.ResNet50,
    }

    if backbone_name not in backbones:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(backbones.keys())}")

    # Load pretrained backbone (frozen)
    base_model = backbones[backbone_name](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze all layers initially
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=f"{backbone_name}_transfer")
    return model, base_model


def unfreeze_top_layers(base_model, num_layers=20):
    """Unfreeze the top N layers of a base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    total = len(base_model.layers)
    print(f"  Fine-tuning: {trainable}/{total} layers trainable")
    return base_model


if __name__ == "__main__":
    print("Model architectures:")
    for name, builder in [
        ("Custom CNN", lambda: (build_custom_cnn(), None)),
        ("MobileNetV2", lambda: build_transfer_model("MobileNetV2")),
        ("EfficientNetB0", lambda: build_transfer_model("EfficientNetB0")),
        ("ResNet50", lambda: build_transfer_model("ResNet50")),
    ]:
        model, _ = builder()
        params = model.count_params()
        print(f"  {name:20s}: {params:>12,} params")
