"""
stage_one_model.py — 4-Block lightweight binary CNN for Stage 1 Gatekeeper.

Aggressive spatial downsampling via 4 MaxPool layers reduces flatten size
and keeps total params under ~540K. Temporal structure is preserved in
the conv feature maps for Stage 2 transfer learning.

Architecture:
  Block 1: Conv2D(32) → BN → ReLU → MaxPool(2,2)
  Block 2: Conv2D(64) → BN → ReLU → MaxPool(2,2)
  Block 3: Conv2D(64) → BN → ReLU → MaxPool(2,2)
  Block 4: Conv2D(64) → BN → ReLU → MaxPool(2,2)
  Flatten → Dense(64, relu) → Dense(32, relu) → Dropout(0.5) → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(config, input_shape: tuple) -> Model:
    """Build a 4-block binary CNN optimized for edge inference.

    Parameters
    ----------
    config      : Config-like object with `dropout_rate` and `learning_rate`
    input_shape : tuple, e.g. (64, 438, 1) → (n_mels, time_frames, 1)

    Returns
    -------
    model : tf.keras.Model  (compiled)
    """
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # ── Block 1 ────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Block 2 ────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Block 3 ────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Block 4 ────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ── Classifier Head ───────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dense(32, activation="relu", name="dense2")(x)
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
