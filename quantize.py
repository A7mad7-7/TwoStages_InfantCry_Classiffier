"""
quantize.py — Post-Training INT8 Quantization (COMMON module).

Converts any .keras model to INT8 .tflite using Post-Training Quantization (PTQ).
Uses a subset of training data as the representative dataset.

This module does NOT import any config directly — it receives config
and data via function parameters, making it stage-agnostic.
"""

import os
from typing import List, Optional

import numpy as np
import tensorflow as tf


def quantize_model(
    config,
    preprocessor=None,
    representative_paths: Optional[List[str]] = None,
) -> str:
    """Convert the saved Keras model to an INT8 TFLite model.

    Uses a small subset of training files as the representative dataset.
    Files are loaded one-at-a-time to avoid RAM overflow.

    Parameters
    ----------
    config               : Config-like object with model_save_path, tflite_save_path
    preprocessor          : AudioPreprocessor (with stats computed)
    representative_paths  : List of file paths for calibration (200 files recommended)

    Returns
    -------
    output_path : str — path to the saved .tflite file
    """
    print(f"[Quantize] Loading model from: {config.model_save_path}")
    model = tf.keras.models.load_model(config.model_save_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # ── Representative Dataset for full INT8 quantization ──────────────
    if representative_paths is not None and preprocessor is not None:
        # Use real data, loaded one file at a time (RAM-safe)
        cal_paths = representative_paths[:200]
        print(f"[Quantize] Using {len(cal_paths)} real files for calibration.")

        def representative_dataset_gen():
            for path in cal_paths:
                spec = preprocessor.process_single_file(path)
                yield [spec[np.newaxis, ...]]  # shape: (1, n_mels, time, 1)
    else:
        # Fallback: dummy data
        input_shape = model.input_shape[1:]
        print(f"[Quantize] No representative data — using dummy data "
              f"with shape (100, {input_shape})")
        dummy = np.random.randn(100, *input_shape).astype(np.float32)

        def representative_dataset_gen():
            for i in range(100):
                yield [dummy[i:i+1]]

    converter.representative_dataset = representative_dataset_gen

    # Request full integer quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # ── Convert & Save ─────────────────────────────────────────────────
    print("[Quantize] Converting to INT8 TFLite …")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(config.tflite_save_path), exist_ok=True)
    with open(config.tflite_save_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(config.tflite_save_path) / 1024
    print(f"[Quantize] INT8 TFLite model saved → {config.tflite_save_path} "
          f"({size_kb:.1f} KB)")
    return config.tflite_save_path
