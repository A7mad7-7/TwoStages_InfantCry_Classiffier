"""
stage_one_pipeline.py — Master orchestrator for Stage 1: Gatekeeper (Binary).

Pipeline phases:
  1. Load file paths & stratified split (70/15/15)
  2. Initialize preprocessor (fixed-range normalization)
  3. Load all spectrograms into NumPy arrays
  4. Build & train 4-block binary CNN
  5. Plot learning curves
  6. Evaluate on unseen Test Set (Confusion Matrix + Classification Report)
  7. Quantize → INT8 TFLite for Raspberry Pi 4
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from stage_one_config import Config
from data_loader import DataLoader
from preprocessor import AudioPreprocessor
from stage_one_trainer import Trainer
from quantize import quantize_model


def main() -> None:
    cfg = Config()

    # ================================================================
    # PHASE 1 — Data Loading & Splitting
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Data Loading & Splitting")
    print("=" * 60)

    loader = DataLoader(cfg)
    file_paths, labels = loader.load_file_paths()
    splits = loader.split_data(file_paths, labels)

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    # ================================================================
    # PHASE 2 — Preprocessor Initialization
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Preprocessor Initialization (Instance Norm)")
    print("=" * 60)

    preprocessor = AudioPreprocessor(cfg)
    print(f"[Pipeline] Expected spectrogram shape: {preprocessor.spectrogram_shape}")

    # ================================================================
    # PHASE 3 — Load Spectrograms into NumPy Arrays
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: Loading Spectrograms into NumPy Arrays")
    print("=" * 60)

    X_train, y_train = loader.load_arrays(train_paths, train_labels, preprocessor)
    X_val, y_val     = loader.load_arrays(val_paths, val_labels, preprocessor)
    X_test, y_test   = loader.load_arrays(test_paths, test_labels, preprocessor)

    print(f"[Pipeline] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[Pipeline] Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"[Pipeline] Test:  X={X_test.shape}, y={y_test.shape}")

    # ================================================================
    # PHASE 4 — Model Architecture & Training
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: Model Architecture & Training")
    print("=" * 60)

    trainer = Trainer(cfg)
    input_shape = preprocessor.spectrogram_shape  # (n_mels, time_frames, 1)
    trainer.build(input_shape)

    # Compute class weights from training labels
    unique_classes = np.unique(y_train.astype(int))
    balanced_weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train.astype(int)
    )
    class_weights = dict(zip(unique_classes.tolist(), balanced_weights))

    print("\n[Pipeline] Class weights (index → weight):")
    for idx in sorted(class_weights):
        print(f"  {idx} ({cfg.index_to_label[idx]:>12s}): "
              f"{class_weights[idx]:.4f}")

    history = trainer.train(X_train, y_train, X_val, y_val, class_weights=class_weights)
    trainer.save()
    trainer.plot_learning_curves(history)

    # ================================================================
    # PHASE 5 — Evaluation on UNSEEN Test Set
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 5: Test Set Evaluation")
    print("=" * 60)

    trainer.evaluate(X_test, y_test, split_name="Test")

    # ================================================================
    # PHASE 6 — Edge Deployment (INT8 Quantization)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 6: Edge Deployment — INT8 Quantization")
    print("=" * 60)

    # Use a subset of training files for calibration
    quantize_model(
        cfg,
        preprocessor=preprocessor,
        representative_paths=train_paths[:200],
    )

    # ================================================================
    # Done
    # ================================================================
    print("\n" + "=" * 60)
    print("  ✅  Stage 1 Pipeline complete!")
    print(f"  Keras model      : {cfg.model_save_path}")
    print(f"  TFLite INT8      : {cfg.tflite_save_path}")
    print(f"  Confusion Matrix : {cfg.confusion_matrix_path}")
    print(f"  Learning Curves  : {cfg.learning_curves_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
