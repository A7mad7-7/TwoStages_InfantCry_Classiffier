"""
data_loader.py — File discovery, stratified splitting, and NumPy array
creation (COMMON module).

Golden Rule: split FIRST, no data leakage between sets.

This module does NOT import any config directly — it receives config
via constructor, making it stage-agnostic.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter


class DataLoader:
    """Handles dataset discovery, splitting, and NumPy array creation."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : any Config-like object with attributes:
            dataset_dir, class_names, train_ratio, val_ratio, test_ratio,
            random_seed, batch_size, n_mels, target_length, hop_length
        """
        self.cfg = config
        np.random.seed(config.random_seed)

    # ── Step 1: Discover file paths ────────────────────────────────────
    def load_file_paths(self) -> Tuple[List[str], List[str]]:
        """Walk dataset directories and return (file_paths, labels).

        Subfolder names map directly to class names.

        Returns
        -------
        file_paths : list[str]  – absolute paths to .wav files
        labels     : list[str]  – corresponding class names
        """
        file_paths: List[str] = []
        labels: List[str] = []

        for class_name in sorted(self.cfg.class_names):
            class_dir = os.path.join(self.cfg.dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[DataLoader] WARNING: directory not found: {class_dir}")
                continue

            files = sorted([
                f for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f))
                and f.lower().endswith(".wav")
            ])
            for fname in files:
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)

        print(f"[DataLoader] Found {len(file_paths)} files across "
              f"{len(set(labels))} classes.")
        self._print_class_distribution(labels, tag="Dataset")
        return file_paths, labels

    # ── Step 2: Stratified Train / Val / Test Split ────────────────────
    def split_data(
        self,
        file_paths: List[str],
        labels: List[str],
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Stratified split into train / val / test (70 / 15 / 15).

        Returns
        -------
        dict with keys 'train', 'val', 'test', each mapping to
        (file_paths_list, labels_list).
        """
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

        print(f"[DataLoader] Loaded arrays: X={X.shape}, y={y.shape}")
        return X, y

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _print_class_distribution(labels: List[str], tag: str = "") -> None:
        dist = Counter(labels)
        parts = [f"{cls}: {cnt}" for cls, cnt in sorted(dist.items())]
        print(f"  [{tag}] {' | '.join(parts)}")
