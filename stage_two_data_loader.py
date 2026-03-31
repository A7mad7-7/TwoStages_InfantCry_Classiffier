"""
stage_two_data_loader.py — File discovery and data loading for Stage 2.

Discovers paths, groups into 3 macro-classes ("Pain", "Hungry", "Tired")
and ignores "noise" and "silence" entirely based on the config map.
"""

import os
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class StageTwoDataLoader:
    """Handles Phase 2 dataset discovery, mapping to macro-classes, and NumPy loading."""

    def __init__(self, config):
        self.cfg = config
        np.random.seed(config.random_seed)

    # ── Step 1: Discover file paths ────────────────────────────────────
    def load_file_paths(self) -> Tuple[List[str], List[str]]:
        """Walk dataset directories, map folders to macro-classes.
        Ignores folders not in `folder_to_class` (like 'noise', 'silence').
        """
        file_paths: List[str] = []
        labels: List[str] = []

        if not os.path.exists(self.cfg.dataset_dir):
            raise FileNotFoundError(f"Directory not found: {self.cfg.dataset_dir}")

        for folder_name in sorted(os.listdir(self.cfg.dataset_dir)):
            class_dir = os.path.join(self.cfg.dataset_dir, folder_name)
            if not os.path.isdir(class_dir):
                continue

            # Determine the macro-class from the folder name
            macro_class = self.cfg.folder_to_class.get(folder_name)
            
            # If not mapped (or mapped to something else), ignore it
            if macro_class not in self.cfg.class_names:
                continue

            files = sorted([
                f for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f))
                and f.lower().endswith(".wav")
            ])
            for fname in files:
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(macro_class)

        print(f"[StageTwoDataLoader] Found {len(file_paths)} files across "
              f"{len(set(labels))} target macro-classes.")
        self._print_class_distribution(labels, tag="Dataset")
        return file_paths, labels

    # ── Step 2: Stratified Split ───────────────────────────────────────
    def split_data(
        self,
        file_paths: List[str],
        labels: List[str],
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Stratified split into train / val / test."""
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
            print(f"[StageTwoDataLoader] {name:>5s} set: {len(paths)} samples")
            self._print_class_distribution(lbls, tag=f"  {name}")
        return splits

    # ── Step 3: Load all files into NumPy arrays ───────────────────────
    def load_arrays(
        self,
        file_paths: List[str],
        labels: List[str],
        preprocessor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process all files and return (X, y) as NumPy arrays.

        Parameters
        ----------
        file_paths   : list of absolute paths to .wav files
        labels       : list of string class labels
        preprocessor : AudioPreprocessor with stats already computed

        Returns
        -------
        X : np.ndarray, shape (N, n_mels, time_frames, 1), dtype float32
        y : np.ndarray, shape (N,), dtype float32
        """
        label_to_idx = self.cfg.label_to_index
        int_labels = np.array([label_to_idx[l] for l in labels], dtype=np.float32)

        specs = []
        for path in tqdm(file_paths, desc="Loading spectrograms", unit="file"):
            spec = preprocessor.process_single_file(path)
            specs.append(spec)

        X = np.array(specs, dtype=np.float32)
        y = int_labels

        print(f"[StageTwoDataLoader] Loaded arrays: X={X.shape}, y={y.shape}")
        return X, y

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _print_class_distribution(labels: List[str], tag: str = "") -> None:
        dist = Counter(labels)
        parts = [f"{cls}: {cnt}" for cls, cnt in sorted(dist.items())]
        print(f"  [{tag}] {' | '.join(parts)}")
