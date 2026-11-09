"""
stage_one_model.py — 4-Block lightweight binary CNN for Stage 1 Gatekeeper.

Uses 4 Conv2D blocks with doubled filter counts, GroupNormalization, and
GlobalAveragePooling2D for time-shift invariance, keeping parameters compact (~375K)
and edge-friendly.

Architecture:
  Block 1: Conv2D(64)  → GroupNorm(8) → ReLU → MaxPool(2,2)
  Block 2: Conv2D(128) → GroupNorm(8) → ReLU → MaxPool(2,2)
  Block 3: Conv2D(128) → GroupNorm(8) → ReLU → MaxPool(2,2)
  Block 4: Conv2D(128) → GroupNorm(8) → ReLU → MaxPool(2,2)
  GlobalAveragePooling2D → Dense(32, relu) → Dropout → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(config, input_shape: tuple) -> Model:
    """Build a 4-block binary CNN optimized for edge inference.

    Parameters
    ----------
    config      : Config-like object with `dropout_rate` and `learning_rate`
    input_shape : tuple, e.g. (64, 626, 1) → (n_mels, time_frames, 1) for 10.0s audio

    Returns
    -------
    model : tf.keras.Model  (compiled)
    """
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # ── Block 1 ────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.GroupNormalization(groups=8, name="gn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Block 2 ────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2")(x)
    x = layers.GroupNormalization(groups=8, name="gn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Block 3 ────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.GroupNormalization(groups=8, name="gn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Block 4 ────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv4")(x)
    x = layers.GroupNormalization(groups=8, name="gn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ── Classifier Head ───────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(config.dropout_rate, name="dropout1")(x)

    # Binary output — sigmoid activation
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Stage1_Gatekeeper")

    # ── Compile ────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model
