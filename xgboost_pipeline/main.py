import os
import numpy as np
import librosa

from xgboost_pipeline.config_xgb import ConfigXGB
from xgboost_pipeline.data_loader_xgb import DataLoaderXGB
from xgboost_pipeline.feature_extractor import FeatureExtractor
from xgboost_pipeline.train_xgb import TrainerXGB

def load_audio_from_paths(paths, sr, target_len):
    """Loads and strictly pads/truncates raw validation/test audio."""
    print(f"Loading {len(paths)} files from disk...")
    out = []
    for i, p in enumerate(paths):
        y, _ = librosa.load(p, sr=sr)
        if len(y) >= target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        out.append(y)
        if (i+1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(paths)}")
    return out

def main():
    cfg = ConfigXGB()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # ── Phase 1: Data
    loader = DataLoaderXGB(cfg)
    paths, labels = loader.load_file_paths()
    splits = loader.split_data(paths, labels)
    
    X_train_paths, y_train_labels = splits["train"]
    X_val_paths, y_val_labels = splits["val"]
    X_test_paths, y_test_labels = splits["test"]
    
    # Encode labels mathematically
    y_val = np.array([cfg.label_to_index[l] for l in y_val_labels])
    y_test = np.array([cfg.label_to_index[l] for l in y_test_labels])
    
    # Augment Training
    X_train_audio, y_train_labels_aug = loader.augment_training_set(X_train_paths, y_train_labels)
    y_train = np.array([cfg.label_to_index[l] for l in y_train_labels_aug])
    
    # Load Val and Test arrays
    X_val_audio = load_audio_from_paths(X_val_paths, cfg.sample_rate, cfg.target_length)
    X_test_audio = load_audio_from_paths(X_test_paths, cfg.sample_rate, cfg.target_length)
    
    # ── Phase 2: Feature Extraction
    extractor = FeatureExtractor(cfg)
    print("\n[Pipeline] Extracting statistical features for Train set...")
    X_train_feat = extractor.process_dataset(X_train_audio)
    print("[Pipeline] Extracting statistical features for Val set...")
    X_val_feat = extractor.process_dataset(X_val_audio)
    print("[Pipeline] Extracting statistical features for Test set...")
    X_test_feat = extractor.process_dataset(X_test_audio)
    
    # ── Phase 3: Normalize & Train
    trainer = TrainerXGB(cfg)
    trainer.fit_scaler(X_train_feat)
    
    X_train_norm = trainer.transform(X_train_feat)
    X_val_norm = trainer.transform(X_val_feat)
    X_test_norm = trainer.transform(X_test_feat)
    
    trainer.train(X_train_norm, y_train, X_val_norm, y_val)
    
    # ── Phase 4: Evaluate & Save
    trainer.evaluate(X_test_norm, y_test)
    trainer.save()
    
    print("\n============================================================")
    print("  ✅  XGBoost Pipeline complete!")
    print(f"  Model  : {cfg.model_save_path}")
    print(f"  Scaler : {cfg.scaler_save_path}")
    print(f"  Matrix : {cfg.confusion_matrix_path}")
    print("============================================================")

if __name__ == "__main__":
    main()
