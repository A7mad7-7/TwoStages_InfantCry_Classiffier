"""
stage_two_pipeline.py — Master orchestrator for Stage 2: Expert Classifier.

Pipeline phases:
  1. Load file paths, map to ['Pain', 'Hungry', 'Tired'], and stratify split
  2. Compute Z-score stats from training files
  3. Load all spectrograms into NumPy arrays
  4. Build transfer learning model (frozen base, new head) & train
  5. Plot learning curves
  6. Evaluate Confusion Matrix on Validation AND Test sets
  7. Quantize → INT8 TFLite for Edge Deployment
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from stage_two_config import Config
from stage_two_data_loader import StageTwoDataLoader
from preprocessor import AudioPreprocessor
from stage_two_trainer import StageTwoTrainer
from quantize import quantize_model


def main() -> None:
    cfg = Config()

    # ================================================================
    # PHASE 1 — Data Loading, Mapping & Splitting
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Data Loading & Splitting (Stage 2)")
    print("=" * 60)

    loader = StageTwoDataLoader(cfg)
    file_paths, labels = loader.load_file_paths()
    
    if len(file_paths) == 0:
        print("\n[ERROR] No data found for Stage 2. Check path:", cfg.dataset_dir)
        return
        
    splits = loader.split_data(file_paths, labels)

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    # ================================================================
    # PHASE 2 — Compute Z-Score Stats
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Computing Z-Score Statistics (Stage 2 Train Data)")
    print("=" * 60)

    preprocessor = AudioPreprocessor(cfg)
    preprocessor.compute_stats_streaming(train_paths)
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
    # PHASE 4 — Transfer Learning Model Architecture & Training
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: Model Architecture & Transfer Learning")
    print("=" * 60)

    trainer = StageTwoTrainer(cfg)
    input_shape = preprocessor.spectrogram_shape
    trainer.build(input_shape)

    # Compute class weights for ['Pain', 'Hungry', 'Tired']
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
    # PHASE 5 — Evaluation on Validation and Unseen Test Set
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 5: Test Set Evaluation")
    print("=" * 60)

    print("\n→ Evaluating on Validation Split:")
    trainer.evaluate(X_val, y_val, split_name="Validation", confusion_matrix_save_path=cfg.confusion_matrix_val_path)
    
    print("\n→ Evaluating on Test Split:")
    trainer.evaluate(X_test, y_test, split_name="Test", confusion_matrix_save_path=cfg.confusion_matrix_test_path)

    # ================================================================
    # PHASE 6 — Edge Deployment (INT8 Quantization)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 6: Edge Deployment — INT8 Quantization")
    print("=" * 60)

    quantize_model(
        cfg,
        preprocessor=preprocessor,
        representative_paths=train_paths[:200],
    )

    # ================================================================
    # Done
    # ================================================================
    print("\n" + "=" * 60)
    print("  ✅  Stage 2 Pipeline complete!")
    print(f"  Keras model      : {cfg.model_save_path}")
    print(f"  TFLite INT8      : {cfg.tflite_save_path}")
    print(f"  CM (Validation)  : {cfg.confusion_matrix_val_path}")
    print(f"  CM (Test)        : {cfg.confusion_matrix_test_path}")
    print(f"  Learning Curves  : {cfg.learning_curves_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
