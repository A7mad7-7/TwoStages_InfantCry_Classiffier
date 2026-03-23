"""
preprocessor.py — Audio-to-spectrogram preprocessing (COMMON module).

Converts raw 16kHz audio → fixed-length → Log-Mel Spectrogram → Z-score normalized.
Reusable by both Stage 1 and Stage 2 pipelines.

CRITICAL: Uses fmax from config to cap frequency range.

This module does NOT import any config directly — it receives config
via constructor, making it stage-agnostic.
"""

from typing import List, Optional

import numpy as np
import librosa
from tqdm import tqdm


class AudioPreprocessor:
    """Stateful preprocessor that stores training-set normalization stats."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : any Config-like object with attributes:
            sample_rate, target_length, n_fft, hop_length, n_mels, fmax
        """
        self.cfg = config
        # Z-score stats — set after calling compute_stats_streaming()
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    # ── Single-File Processing ─────────────────────────────────────────
    def load_and_resample(self, path: str) -> np.ndarray:
        """Load an audio file and resample to the target sample rate."""
        audio, _ = librosa.load(path, sr=self.cfg.sample_rate)
        return audio

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate to exactly `target_length` samples."""
        target = self.cfg.target_length
        if len(audio) >= target:
            return audio[:target]
        pad_width = target - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant")

    def to_log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Convert a 1-D audio array to a Log-Mel Spectrogram.

        Returns
        -------
        log_mel : np.ndarray, shape (n_mels, time_frames)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmax=self.cfg.fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel

    # ── Streaming Z-Score Stats (RAM-safe) ─────────────────────────────
    def compute_stats_streaming(self, file_paths: List[str]) -> None:
        """Compute global mean & std from training files WITHOUT loading all
        spectrograms into RAM. Processes one file at a time.

        Uses the numerically stable two-pass online formula:
          mean = sum(x) / N
          std  = sqrt( sum(x²)/N - mean² )
        """
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        for path in tqdm(file_paths, desc="Computing Z-score stats", unit="file"):
            audio = self.load_and_resample(path)
            audio = self.pad_or_truncate(audio)
            spec = self.to_log_mel_spectrogram(audio)
            total_sum += np.sum(spec)
            total_sq_sum += np.sum(spec ** 2)
            total_count += spec.size

        self.mean = total_sum / total_count
        variance = (total_sq_sum / total_count) - (self.mean ** 2)
        self.std = np.sqrt(max(variance, 0.0))
        if self.std < 1e-8:
            self.std = 1e-8

        print(f"[Preprocessor] Training stats  →  mean={self.mean:.4f}, "
              f"std={self.std:.4f}")

    def normalize(self, spectrogram: np.ndarray) -> np.ndarray:
        """Z-score normalize using precomputed training-set statistics."""
        assert self.mean is not None, "Call compute_stats_streaming() first!"
        return (spectrogram - self.mean) / self.std

    # ── Full single-file pipeline (used by tf.data) ────────────────────
    def process_single_file(self, path: str) -> np.ndarray:
        """Load → pad/truncate → mel spec → normalize → add channel dim.

        Returns
        -------
        spec : np.ndarray, shape (n_mels, time_frames, 1), dtype float32
        """
        audio = self.load_and_resample(path)
        audio = self.pad_or_truncate(audio)
        spec = self.to_log_mel_spectrogram(audio)
        spec = self.normalize(spec)
        return spec[..., np.newaxis].astype(np.float32)

    @property
    def spectrogram_shape(self) -> tuple:
        """Compute the expected spectrogram output shape from config."""
        time_frames = 1 + (self.cfg.target_length // self.cfg.hop_length)
        return (self.cfg.n_mels, time_frames, 1)
