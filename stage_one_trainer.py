"""
stage_one_trainer.py — Training loop, learning curves, and test-set evaluation
for the Stage 1 Gatekeeper (binary classifier).

Accepts plain NumPy arrays for training and evaluation.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage_one_model import build_model


class Trainer:
    """Encapsulates model building, training, and evaluation for Stage 1."""

    def __init__(self, config):
        self.cfg = config
        self.model: tf.keras.Model = None

    # ── Build ──────────────────────────────────────────────────────────
    def build(self, input_shape: tuple) -> None:
        """Construct and compile the model."""
        self.model = build_model(self.cfg, input_shape)
        self.model.summary()

    # ── Train (accepts NumPy arrays) ───────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> tf.keras.callbacks.History:
        """Run the training loop with EarlyStopping + ReduceLROnPlateau.

        Parameters
        ----------
        X_train      : np.ndarray, shape (N, n_mels, time_frames, 1)
        y_train      : np.ndarray, shape (N,)
        X_val        : np.ndarray, shape (M, n_mels, time_frames, 1)
        y_val        : np.ndarray, shape (M,)
        class_weights: dict {class_index: weight} or None
        """
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.cfg.reduce_lr_factor,
                patience=self.cfg.reduce_lr_patience,
                min_lr=self.cfg.min_lr,
                verbose=1,
            ),
        ]

        if class_weights is not None:
            print(f"[Trainer] Using class_weights: {class_weights}")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
        return history

    # ── Save Model ─────────────────────────────────────────────────────
    def save(self) -> str:
        """Save the trained model."""
        os.makedirs(os.path.dirname(self.cfg.model_save_path), exist_ok=True)
        self.model.save(self.cfg.model_save_path)
        print(f"[Trainer] Model saved → {self.cfg.model_save_path}")
        return self.cfg.model_save_path

    # ── Learning Curves ────────────────────────────────────────────────
    def plot_learning_curves(self, history: tf.keras.callbacks.History) -> None:
        """Plot training vs validation accuracy and loss."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
        ax1.set_title("Stage 1 — Accuracy", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
        ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
        ax2.set_title("Stage 1 — Loss", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(self.cfg.learning_curves_path), exist_ok=True)
        fig.savefig(self.cfg.learning_curves_path, dpi=150)
        plt.close(fig)
        print(f"[Trainer] Learning curves saved → {self.cfg.learning_curves_path}")

    # ── Evaluate on NumPy arrays ───────────────────────────────────────
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "Test",
    ) -> None:
        """Compute predictions and print classification report + confusion matrix.

        Parameters
        ----------
        X          : np.ndarray, shape (N, n_mels, time_frames, 1)
        y          : np.ndarray, shape (N,)
        split_name : str — displayed in the report header
        """
        probs = self.model.predict(X, batch_size=self.cfg.batch_size, verbose=0).flatten()
        y_pred = (probs >= 0.5).astype(int)
        y_true = y.astype(int)

        # Classification Report
        target_names = self.cfg.class_names
        print(f"\n{'=' * 60}")
        print(f"  Classification Report — {split_name} Set")
        print(f"{'=' * 60}")
        print(classification_report(y_true, y_pred, target_names=target_names,
                                    zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, target_names, split_name)

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list,
        title: str,
    ) -> None:
        """Plot and save a styled confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=f"Confusion Matrix — {title} Set (Stage 1)",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16,
                )

        fig.tight_layout()
        os.makedirs(os.path.dirname(self.cfg.confusion_matrix_path), exist_ok=True)
        fig.savefig(self.cfg.confusion_matrix_path, dpi=150)
        plt.close(fig)
        print(f"[Trainer] Confusion matrix saved → {self.cfg.confusion_matrix_path}")
