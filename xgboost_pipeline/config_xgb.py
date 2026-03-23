import os
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ConfigXGB:
    """Immutable configuration for the XGBoost ML pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = os.path.join(base_dir, "Baby Crying Sounds")
    output_dir: str = os.path.join(base_dir, "xgboost_pipeline", "output")

    # ── Audio Parameters ───────────────────────────────────────────────
    sample_rate: int = 8_000           # Target sample rate in Hz
    duration_sec: float = 7.0          # Fixed clip length
    target_length: int = 56_000        # sample_rate * duration_sec

    # ── Feature Extraction ─────────────────────────────────────────────
    n_mfcc: int = 13                   # Number of MFCCs to extract
    n_fft: int = 1024
    hop_length: int = 256

    # ── Splitting ──────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Augmentation (No Pitch Shifting) ───────────────────────────────
    target_samples_per_class: int = 300
    time_stretch_range: tuple = (0.8, 1.3)
    noise_snr_range: tuple = (10, 15)      # dB

    # ── Class Mapping (4 Macro Classes) ────────────────────────────────
    folder_to_class: Dict[str, str] = field(
        default_factory=lambda: {
            "belly_pain": "Pain",
            "discomfort": "Pain",
            "cold_hot": "Pain",
            "hungry": "Hungry",
            "tired": "Tired",
            "burping": "Tired",
            "laugh": "Tired",
            "noise": "No_Response",
            "silence": "No_Response"
        }
    )

    class_names: List[str] = field(
        default_factory=lambda: [
            "Hungry",
            "No_Response",
            "Pain",
            "Tired",
        ]
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
        return os.path.join(self.output_dir, "xgboost_model.json")

    @property
    def scaler_save_path(self) -> str:
        return os.path.join(self.output_dir, "scaler.pkl")

    @property
    def confusion_matrix_path(self) -> str:
        return os.path.join(self.output_dir, "confusion_matrix_xgb.png")
