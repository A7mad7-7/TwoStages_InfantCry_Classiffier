"""
preprocessor.py — Audio-to-spectrogram preprocessing (COMMON module).

Converts raw 16kHz audio → fixed-length → Log-Mel Spectrogram → Fixed-range [0,1] normalized.
Reusable by both Stage 1 and Stage 2 pipelines.

CRITICAL: Uses fmax from config to cap frequency range.

This module does NOT import any config directly — it receives config
via constructor, making it stage-agnostic.
"""

# from typing import List, Optional

import numpy as np
import librosa
# from tqdm import tqdm


class AudioPreprocessor:
    """Stateless preprocessor with fixed-range normalization (no training stats needed)."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : any Config-like object with attributes:
            sample_rate, target_length, n_fft, hop_length, n_mels, fmax
        """
        self.cfg = config

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
        # Absolute scaling: ref=1.0, top_db=None prevents dynamic instance thresholding
        log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
        return log_mel

    def normalize(self, spectrogram: np.ndarray) -> np.ndarray:
        """Fixed-range absolute normalization: [-100, 55] dB → [0, 1].

        Unlike relative normalization, this preserves absolute energy differences:
          - Pure silence (1e-10 amplitude) stays near 0 (-100 dB → 0.0)
          - Loud background noise reaches ~0.45 (~15 dB)
          - Screaming cries reach ~0.8+ (~20-55 dB)
        """
        # Range is 155 dB total (-100 dB to +55 dB)
        return np.clip((spectrogram + 100.0) / 155.0, 0.0, 1.0)

    # ── Full single-file pipeline ──────────────────────────────────────
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
