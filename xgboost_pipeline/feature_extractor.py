import numpy as np
import librosa
from xgboost_pipeline.config_xgb import ConfigXGB

class FeatureExtractor:
    """Extracts classical statistical audio features for ML modeling."""

    def __init__(self, config: ConfigXGB):
        self.cfg = config

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs, Spectral Centroid, and ZCR.
        Calculate Mean, Std, Min, Max for each coefficient over time.
        Returns a flat 1D array of shape (num_features,).
        """
        sr = self.cfg.sample_rate

        # 1. MFCCs (shape: n_mfcc x frames)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)

        # 2. Spectral Centroid (shape: 1 x frames)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr,
            n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        cent_mean = np.mean(centroid, axis=1)
        cent_std = np.std(centroid, axis=1)
        cent_min = np.min(centroid, axis=1)
        cent_max = np.max(centroid, axis=1)

        # 3. Zero Crossing Rate (shape: 1 x frames)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, hop_length=self.cfg.hop_length
        )
        zcr_mean = np.mean(zcr, axis=1)
        zcr_std = np.std(zcr, axis=1)
        zcr_min = np.min(zcr, axis=1)
        zcr_max = np.max(zcr, axis=1)

        # 4. Concatenate all into a single 1D feature vector
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_min, mfcc_max,
            cent_mean, cent_std, cent_min, cent_max,
            zcr_mean, zcr_std, zcr_min, zcr_max
        ])
        
        return features

    def process_dataset(self, audio_list: list[np.ndarray]) -> np.ndarray:
        """Process a list of audio arrays into a 2D feature matrix."""
        all_features = []
        for i, audio in enumerate(audio_list):
            feats = self.extract_features(audio)
            all_features.append(feats)
            if (i + 1) % 500 == 0:
                print(f"  [FeatureExtractor] Processed {i+1}/{len(audio_list)}")
        
        matrix = np.array(all_features)
        print(f"[FeatureExtractor] Extracted {matrix.shape[1]} features for {matrix.shape[0]} samples.")
        return matrix
