#!/usr/bin/env python3
"""
evaluate_stage_two.py

Standalone script to evaluate the Stage 2 (Expert) model on the test set.
It loads the test data (15% split as defined in stage_two_data_loader.py), 
evaluates the saved .keras model, prints the Test Loss and Accuracy, 
and generates a Confusion Matrix.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from stage_two_config import Config
from stage_two_data_loader import StageTwoDataLoader
from preprocessor import AudioPreprocessor

def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, save_path: str):
    """Plot and save a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Confusion Matrix saved → {save_path}")


def main():
    cfg = Config()
    
    print("=" * 60)
    print("  Evaluating Stage 2 (Expert) Model on Test Set")
    print("=" * 60)

    # 1. Load Data Splits
    loader = StageTwoDataLoader(cfg)
    file_paths, labels = loader.load_file_paths()
    splits = loader.split_data(file_paths, labels)
    test_paths, test_labels = splits["test"]

    # 2. Preprocess Test Set
    preprocessor = AudioPreprocessor(cfg)
    X_test, y_test = loader.load_arrays(test_paths, test_labels, preprocessor)
    
    # 3. Load Saved Model
    model_path = cfg.model_save_path
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}. Please train Stage 2 first.")
        return
        
    print(f"\n[Evaluate] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Evaluate Test Loss and Accuracy
    print("\n[Evaluate] Running evaluation on Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=cfg.batch_size, verbose=1)
    print(f"\n✅ Final Test Loss     : {test_loss:.4f}")
    print(f"✅ Final Test Accuracy : {test_acc*100:.2f}%")
    
    # 5. Generate Predictions and Confusion Matrix
    probs = model.predict(X_test, batch_size=cfg.batch_size, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = y_test.astype(int)
    
    # Classification Report
    target_names = cfg.class_names
    print(f"\n{'=' * 60}")
    print(f"  Classification Report — Test Set")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    save_path = os.path.join("Stage_two_output", "standalone_test_confusion_matrix.png")
    plot_confusion_matrix(cm, target_names, "Stage 2 — Test Set Confusion Matrix", save_path)


if __name__ == "__main__":
    main()

