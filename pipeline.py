"""
pipeline.py — Master orchestrator for the Infant Cry Classifier.

Runs all phases in sequence:
  1. Load & split data
  2. Augment training set
  3. Preprocess → spectrograms
  4. Build & train model
  5. Evaluate on validation set
  6. Quantize → INT8 TFLite
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import Config
from data_loader import DataLoader
from preprocessor import AudioPreprocessor
from train import Trainer
from quantize import quantize_model


def main() -> None:
    # ── Configuration ──────────────────────────────────────────────────
    cfg = Config()

    # ================================================================
    # PHASE 1 — Data Preparation
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Data Preparation")
    print("=" * 60)

    loader = DataLoader(cfg)

    # 1a. Discover files and merge noise + silence → no_response
    file_paths, labels = loader.load_file_paths()

    # 1b. Stratified split (70 / 15 / 15) — BEFORE augmentation
    splits = loader.split_data(file_paths, labels)
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    # 1c. Augment ONLY the training set → balance to 300 per class
    train_audio, train_labels_aug = loader.augment_training_set(
        train_paths, train_labels
    )
    # train_audio: list of np.ndarray (raw audio arrays)
    # train_labels_aug: list of str

    # ================================================================
    # PHASE 2 — Preprocessing (Audio → Mel Spectrogram)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Preprocessing Pipeline")
    print("=" * 60)

    preprocessor = AudioPreprocessor(cfg)

    # Process training set from in-memory arrays (includes augmented)
    print("[Pipeline] Processing training audio arrays …")
    X_train_specs = preprocessor.process_from_arrays(train_audio)
    # shape: (N_train, 64, 157)

    # Compute Z-score stats from training spectrograms ONLY
    preprocessor.compute_stats(X_train_specs)

    # Process val & test from file paths
    print("[Pipeline] Processing validation files …")
    X_val_specs = preprocessor.process_from_paths(val_paths)
    print("[Pipeline] Processing test files …")
    X_test_specs = preprocessor.process_from_paths(test_paths)

    # Normalize all sets using training stats & add channel dim
    X_train = preprocessor.finalize(X_train_specs)  # (N, 64, 157, 1)
    X_val = preprocessor.finalize(X_val_specs)
    X_test = preprocessor.finalize(X_test_specs)

    # Convert string labels → integer indices
    y_train = np.array([cfg.label_to_index[l] for l in train_labels_aug])
    y_val = np.array([cfg.label_to_index[l] for l in val_labels])
    y_test = np.array([cfg.label_to_index[l] for l in test_labels])

    print(f"\n[Pipeline] Final shapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # ================================================================
    # PHASE 3 + 4 — Model Architecture & Training
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3 & 4: Model Architecture & Training")
    print("=" * 60)

    trainer = Trainer(cfg)
    input_shape = X_train.shape[1:]  # (n_mels, time_frames, 1)
    trainer.build(input_shape)

    # ── Compute class weights ──────────────────────────────────────
    # Using sklearn's balanced weights (inversely proportional to class frequency)
    unique_classes = np.unique(y_train)
    balanced_weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train
    )
    class_weights = dict(zip(unique_classes.tolist(), balanced_weights))

    print("\n[Pipeline] Class weights (index → weight):")
    for idx in sorted(class_weights):
        print(f"  {idx} ({cfg.index_to_label[idx]:>12s}): {class_weights[idx]:.4f}")

    trainer.train(X_train, y_train, X_val, y_val, class_weights=class_weights)

    # Save the trained model
    trainer.save()

    # Evaluate on validation set
    trainer.evaluate(X_val, y_val, split_name="Validation")

    # ================================================================
    # PHASE 5 — Edge Deployment (INT8 Quantization)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 5: Edge Deployment — INT8 Quantization")
    print("=" * 60)

    # Use a subset of training data as representative dataset
    quantize_model(cfg, representative_data=X_train[:200])

    # ================================================================
    # Done
    # ================================================================
    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete!")
    print(f"  Keras model : {cfg.model_save_path}")
    print(f"  TFLite INT8 : {cfg.tflite_save_path}")
    print(f"  Confusion   : {cfg.confusion_matrix_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
