"""
stage_two_config.py — Configuration for Stage 2: The Expert Classifier.

Multi-class Transfer Learning task: Pain vs Hungry vs Tired.
All hyperparameters, paths, and class mappings for Stage 2 live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import os


@dataclass
class Config:
    """Configuration for the Stage 2 Transfer Learning pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    dataset_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "clean_data", "Stage_Two_Data"
    )
    output_dir: str = "/content/drive/MyDrive/code/Stage_two_output" if os.path.exists("/content/drive/MyDrive/code") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Stage_two_output"
    )
    stage1_model_path: str = "/content/drive/MyDrive/code/Stage_one_output/stage1_gatekeeper.keras" if os.path.exists("/content/drive/MyDrive/code") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Stage_one_output", "stage1_gatekeeper.keras"
    )

    # ── Audio Parameters (MUST Match Stage 1) ──────────────────────────
    sample_rate: int = 16_000
    duration_sec: float = 7.0
    target_length: int = 112_000

    # ── Mel-Spectrogram (MUST Match Stage 1) ───────────────────────────
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    fmax: int = 4000

    # ── Splitting ──────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Training ───────────────────────────────────────────────────────
    learning_rate: float = 1e-5        # Lower LR for fine-tuning
    batch_size: int = 16
    epochs: int = 80
    early_stop_patience: int = 10
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    # ── Class Mapping (Stage 2 Macro-classes) ──────────────────────────
    class_names: List[str] = field(
        default_factory=lambda: ["Pain", "Hungry", "Tired"]
    )

    folder_to_class: Dict[str, str] = field(
        default_factory=lambda: {
            "belly pain": "Pain",
            "discomfort": "Pain",
            "cold_hot": "Pain",
            "hungry": "Hungry",
            "tired": "Tired",
            "burping": "Tired",
            "laugh": "Tired",
            # Note: 'noise' and 'silence' are intentionally omitted
            # so they are ignored by the loader.
        }
    )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def label_to_index(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names)}

    @property
    def index_to_label(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.class_names)}

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.output_dir, "stage2_expert.keras")

    @property
    def tflite_save_path(self) -> str:
        return os.path.join(self.output_dir, "stage2_expert.tflite")

    @property
    def confusion_matrix_val_path(self) -> str:
        return os.path.join(self.output_dir, "confusion_matrix_val_stage2.png")

    @property
    def confusion_matrix_test_path(self) -> str:
        return os.path.join(self.output_dir, "confusion_matrix_test_stage2.png")

    @property
    def learning_curves_path(self) -> str:
        return os.path.join(self.output_dir, "learning_curves_stage2.png")
