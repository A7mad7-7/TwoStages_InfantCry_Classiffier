"""
stage_two_model.py — Transfer Learning Model for Stage 2 Expert Classifier.

Loads the saved stage1_gatekeeper.keras model.
Freezes the base convolutional layers (Conv2D & BatchNormalization).
Removes the old binary head and attaches a new 3-class classification head:
  Flatten → Dense(64) → Dropout(0.4) → Dense(32) → Dropout(0.2) → Dense(3, linear)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import os

def build_transfer_model(config, input_shape: tuple) -> Model:
    """Build the Stage 2 Transfer Learning model.

    Parameters
    ----------
    config      : Config-like object with paths, dropouts, and learning rates.
    input_shape : tuple, e.g. (64, 438, 1) → (n_mels, time_frames, 1)

    Returns
    -------
    model : tf.keras.Model  (compiled)
    """
    if not os.path.exists(config.stage1_model_path):
        raise FileNotFoundError(
            f"Stage 1 base model not found at {config.stage1_model_path}. "
            f"Please run Stage 1 pipeline first."
        )

    print(f"[Model] Loading base model from {config.stage1_model_path}")
    base_model = tf.keras.models.load_model(config.stage1_model_path, compile=False)

    # We want to extract the output just before the Flatten layer.
    # In the stage 1 model, the last pool layer is named 'pool4'
    try:
        last_conv_layer = base_model.get_layer("pool4")
        conv_output = last_conv_layer.output
    except ValueError:
        # Fallback if names changed: Find the last MaxPooling2D
        pool_layers = [l for l in base_model.layers if isinstance(l, layers.MaxPooling2D)]
        if not pool_layers:
            raise ValueError("Could not find MaxPooling2D layers in base model.")
        last_conv_layer = pool_layers[-1]
        conv_output = last_conv_layer.output

    # Create a new base model that ends at the last pooling layer
    feature_extractor = Model(inputs=base_model.input, outputs=conv_output, name="Stage1_Feature_Extractor")

    # ── FREEZE THE BASE LAYERS ─────────────────────────────────────────
    # We freeze Block 1, 2, and 3. We UNFREEZE Block 4 ('conv4' and 'bn4') 
    # to fine-tune the deepest spatial features for Stage 2.
    frozen_count = 0
    unfrozen_count = 0
    for layer in feature_extractor.layers:
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization)):
            if "4" in layer.name:  # 'conv4', 'bn4', etc.
                layer.trainable = True
                unfrozen_count += 1
            else:
                layer.trainable = False
                frozen_count += 1
            
    print(f"[Model] Froze {frozen_count} Conv2D/BN layers from blocks 1-3.")
    print(f"[Model] Unfroze {unfrozen_count} Conv2D/BN layers from block 4.")

    # ── ATTACH NEW STAGE 2 HEAD ────────────────────────────────────────
    x = layers.Flatten(name="stage2_flatten")(feature_extractor.output)
    x = layers.Dense(64, activation="relu", name="stage2_dense1")(x)
    x = layers.Dropout(0.4, name="stage2_dropout1")(x)
    x = layers.Dense(32, activation="relu", name="stage2_dense2")(x)
    x = layers.Dropout(0.2, name="stage2_dropout2")(x)
    
    # 3-class output (linear for from_logits=True)
    outputs = layers.Dense(config.num_classes, activation="linear", name="stage2_logits")(x)

    model = Model(inputs=feature_extractor.input, outputs=outputs, name="Stage2_Expert")

    # ── Compile ────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
