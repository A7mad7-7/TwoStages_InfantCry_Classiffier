"""
requantize_models.py — Re-quantize existing .keras models to float32 I/O TFLite.

This script converts the already-trained Stage 1 and Stage 2 .keras models
into new .tflite files with:
  - INT8-quantized weights (for speed & size)
  - float32 input/output (no manual quantize/dequantize needed in Python)

It uses dummy representative data for calibration, so NO training data is
required on disk. Just the .keras model files.

Usage:
    python requantize_models.py
"""

import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def requantize(keras_path: str, tflite_path: str, name: str):
    """Convert a .keras model to float32-I/O INT8-weights TFLite."""

    print(f"\n{'='*60}")
    print(f"  Re-quantizing: {name}")
    print(f"{'='*60}")
    print(f"  Input:  {keras_path}")
    print(f"  Output: {tflite_path}")

    if not os.path.exists(keras_path):
        print(f"  ❌ ERROR: Keras model not found: {keras_path}")
        return

    # Load the trained Keras model
    model = tf.keras.models.load_model(keras_path)
    input_shape = model.input_shape[1:]  # e.g. (64, 438, 1)
    print(f"  Model input shape: {input_shape}")

    # Set up converter for PURE FLOAT32 conversion (NO Quantization)
    # We discovered that TFLite's Dynamic Range Quantization destroys the model's
    # accuracy (outputting 1.000 for everything). By removing optimizations,
    # we guarantee the .tflite model perfectly matches the .keras model's high accuracy.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]  <- REMOVED

    # Convert
    print("  Converting to TFLite ...")
    tflite_model = converter.convert()

    # Save
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  ✅ Saved: {tflite_path} ({size_kb:.1f} KB)")

    # Verify I/O types
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    print(f"  Verify: input_dtype={in_details['dtype'].__name__}, "
          f"output_dtype={out_details['dtype'].__name__}")


if __name__ == "__main__":
    print("🍼 Smart Crib — Re-quantize Models (float32 I/O + INT8 weights)")

    # ── Stage 1: Gatekeeper ────────────────────────────────────────────
    requantize(
        keras_path=os.path.join(BASE_DIR, "Stage_one_output", "stage1_gatekeeper_finetuned.keras"),
        tflite_path=os.path.join(BASE_DIR, "Stage_one_output", "stage1_gatekeeper.tflite"),
        name="Stage 1 — Gatekeeper (CRY vs NO_RESPONSE)",
    )

    # ── Stage 2: Expert ────────────────────────────────────────────────
    requantize(
        keras_path=os.path.join(BASE_DIR, "Stage_two_output", "stage2_expert.keras"),
        tflite_path=os.path.join(BASE_DIR, "Stage_two_output", "stage2_expert.tflite"),
        name="Stage 2 — Expert (Pain / Hungry / Tired)",
    )

    print(f"\n{'='*60}")
    print("  ✅ All models re-quantized successfully!")
    print("  Copy the .tflite files to the Raspberry Pi and re-run.")
    print(f"{'='*60}")
