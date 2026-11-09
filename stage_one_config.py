"""
stage_one_config.py — Configuration for Stage 1: Gatekeeper (Binary Classifier).

Binary task: CRY vs NO_RESPONSE.
All hyperparameters, paths, and class mappings for Stage 1 live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import os


@dataclass
class Config:
    """Configuration for the Stage 1 binary pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    dataset_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "clean_data", "Stage_One_Data"
    )
    cry_dataset_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Data_set", "audio"
    )
    no_response_dataset_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "clean_data", "Stage_One_Data", "NO_RESPONSE"
    )
    yamnet_filter_suffix: str = "_yamnet75"

    output_dir: str = "/content/drive/MyDrive/code/Stage_one_output" if os.path.exists("/content/drive/MyDrive/code") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Stage_one_output"
    )

    # ── NO_RESPONSE Group Splits (Prevent Data Leakage across sessions) ──
    train_video_groups: List[str] = field(default_factory=lambda: [
        "BILL_GATES_REVEALS_I_Was_Wrong_About_Success_128KBPS",
        "Best_Of_Andrew_Wilson_On_The_Whatever_Podcast_2024_128KBPS",
        "How_to_Communicate_With_Confidence_Ease_From_Harvard_Business_School_s_1_Professor_128KBPS",
        "Let_it_all_out.._The_Group_Chat_Podcast_130_128KBPS",
        "man",
        "dude_miaowing",
        "white_noise",
        "collected_background_noise",
        "collected_background_noise1",
    ])
    val_video_groups: List[str] = field(default_factory=lambda: [
        "womens",
        "exercise_bike",
        "running_tap",
    ])
    test_video_groups: List[str] = field(default_factory=lambda: [
        "Confidence_isn_t_loud_here_s_how_to_find_your_voice_Nimi_Mehta_128KBPS",
        "man-woman",
        "doing_the_dishes",
        "pink_noise",
    ])

    # ── Audio Parameters ───────────────────────────────────────────────
    sample_rate: int = 16_000           # Target sample rate in Hz
    duration_sec: float = 10.0          # Fixed clip length in seconds (10s)
    target_length: int = 160_000        # sample_rate * duration_sec (160k samples)

    # ── Mel-Spectrogram Parameters ─────────────────────────────────────
    n_mels: int = 64                    # Mel-frequency bins
    n_fft: int = 1024                   # FFT window size
    hop_length: int = 256               # STFT hop length
    fmax: int = 4000                    # Max frequency — prevents mic-quality overfitting

    # ── Splitting ──────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Training ───────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 50
    early_stop_patience: int = 10
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    dropout_rate: float = 0.5

    # ── Class Mapping (Binary) ─────────────────────────────────────────
    class_names: List[str] = field(
        default_factory=lambda: ["NO_RESPONSE", "CRY"]
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
        return os.path.join(self.output_dir, "stage1_gatekeeper.keras")

    @property
    def tflite_save_path(self) -> str:
        return os.path.join(self.output_dir, "stage1_gatekeeper.tflite")

    @property
    def confusion_matrix_path(self) -> str:
        return os.path.join(self.output_dir, "confusion_matrix_stage1.png")

    @property
    def learning_curves_path(self) -> str:
        return os.path.join(self.output_dir, "learning_curves_stage1.png")
