import os
import random
from typing import List, Tuple, Dict

import numpy as np
import librosa
from sklearn.model_selection import train_test_split

from xgboost_pipeline.config_xgb import ConfigXGB

class DataLoaderXGB:
    """Handles dataset discovery, 4-class splitting, and safe augmentation."""

    def __init__(self, config: ConfigXGB):
        self.cfg = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def load_file_paths(self) -> Tuple[List[str], List[str]]:
        file_paths: List[str] = []
        labels: List[str] = []

        for folder_name in sorted(os.listdir(self.cfg.dataset_dir)):
            folder_path = os.path.join(self.cfg.dataset_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            canonical = folder_name.replace(" ", "_")
            canonical = self.cfg.folder_to_class.get(canonical, canonical)
            
            # Skip if 200 hungry files were deleted previously, we will only read what exists
            if canonical not in self.cfg.class_names:
                continue

            for fname in sorted(os.listdir(folder_path)):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    file_paths.append(fpath)
                    labels.append(canonical)

        print(f"[DataLoader] Found {len(file_paths)} files mapped to 4 Macro-Classes.")
        self._print_class_distribution(labels, tag="Original")
        return file_paths, labels

    def split_data(
        self, file_paths: List[str], labels: List[str],
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        X_train, X_temp, y_train, y_temp = train_test_split(
            file_paths, labels,
            test_size=(1 - self.cfg.train_ratio),
            stratify=labels,
            random_state=self.cfg.random_seed,
        )

        val_fraction = self.cfg.val_ratio / (self.cfg.val_ratio + self.cfg.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_fraction),
            stratify=y_temp,
            random_state=self.cfg.random_seed,
        )

        splits = {
            "train": (X_train, y_train),
            "val":   (X_val,   y_val),
            "test":  (X_test,  y_test),
        }
        for name, (paths, lbls) in splits.items():
            print(f"[DataLoader] {name:>5s} set: {len(paths)} samples")
            self._print_class_distribution(lbls, tag=f"  {name}")
        return splits

    def augment_training_set(
        self, train_paths: List[str], train_labels: List[str],
    ) -> Tuple[List[np.ndarray], List[str]]:
        target = self.cfg.target_samples_per_class
        sr = self.cfg.sample_rate

        class_paths: Dict[str, List[str]] = {}
        for path, label in zip(train_paths, train_labels):
            class_paths.setdefault(label, []).append(path)

        all_audio: List[np.ndarray] = []
        all_labels: List[str] = []

        for cls_name in self.cfg.class_names:
            paths = class_paths.get(cls_name, [])
            print(f"[Augment] {cls_name}: {len(paths)} originals → {target}")

            originals: List[np.ndarray] = []
            for p in paths:
                audio, _ = librosa.load(p, sr=sr)
                # Pad/truncate to strict length for consistency before feature extraction
                audio = self._pad_or_truncate(audio)
                originals.append(audio)
                all_audio.append(audio)
                all_labels.append(cls_name)

            num_needed = target - len(originals)
            if num_needed <= 0:
                if len(originals) > target:
                    all_audio = all_audio[:-(len(originals) - target)]
                    all_labels = all_labels[:-(len(originals) - target)]
                continue

            for i in range(num_needed):
                src = originals[i % len(originals)]
                aug = self._safe_augmentation(src, sr)
                aug = self._pad_or_truncate(aug)
                all_audio.append(aug)
                all_labels.append(cls_name)

        print(f"[Augment] Total training samples: {len(all_audio)}")
        return all_audio, all_labels

    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        target = self.cfg.target_length
        if len(audio) >= target:
            return audio[:target]
        return np.pad(audio, (0, target - len(audio)), mode="constant")

    def _safe_augmentation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applies Time Stretch or Additive Noise. No Pitch Shift."""
        aug = audio.copy()
        transforms = [self._time_stretch, self._add_noise]
        # Always pick 1 random transform, or both
        k = random.randint(1, len(transforms))
        chosen = random.sample(transforms, k)
        for fn in chosen:
            aug = fn(aug, sr)
        return aug

    def _time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        lo, hi = self.cfg.time_stretch_range
        rate = np.random.uniform(lo, hi)
        return librosa.effects.time_stretch(audio, rate=rate)

    def _add_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        lo, hi = self.cfg.noise_snr_range
        snr_db = np.random.uniform(lo, hi)
        sig_power = np.mean(audio ** 2)
        if sig_power < 1e-10:
            return audio
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
        return audio + noise.astype(audio.dtype)

    @staticmethod
    def _print_class_distribution(labels: List[str], tag: str = "") -> None:
        from collections import Counter
        dist = Counter(labels)
        parts = [f"{cls}: {cnt}" for cls, cnt in sorted(dist.items())]
        print(f"  [{tag}] {' | '.join(parts)}")
