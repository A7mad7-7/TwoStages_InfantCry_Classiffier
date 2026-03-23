"""
data_loader.py — File discovery, stratified splitting, and tf.data.Dataset
creation (COMMON module).

Golden Rule: split FIRST, no data leakage between sets.

This module does NOT import any config directly — it receives config
via constructor, making it stage-agnostic.
"""

import os
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataLoader:
    """Handles dataset discovery, splitting, and tf.data.Dataset creation."""

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

    # ── Step 3: tf.data.Dataset Creation (RAM-safe) ────────────────────
    def create_tf_dataset(
        self,
        file_paths: List[str],
        labels: List[str],
        preprocessor,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset that yields (spectrogram, label) batches.

        Each file is loaded and preprocessed ON-THE-FLY via tf.numpy_function,
        so only one batch (~32 spectrograms) is ever in RAM at once.

        Parameters
        ----------
        file_paths   : list of absolute paths to .wav files
        labels       : list of string class labels
        preprocessor : AudioPreprocessor with stats already computed
        shuffle      : whether to shuffle at the start of each epoch

        Returns
        -------
        dataset : tf.data.Dataset yielding (spec, label) batches
                  spec shape: (batch, n_mels, time_frames, 1)
                  label shape: (batch,) float32
        """
        # Convert string labels to integer indices
        label_to_idx = self.cfg.label_to_index
        int_labels = [label_to_idx[l] for l in labels]

        # Expected spectrogram shape (from config)
        spec_shape = preprocessor.spectrogram_shape  # (n_mels, time_frames, 1)

        def _process_file(path_tensor, label_tensor):
            """Wrapper for tf.numpy_function."""
            def _load(path_bytes, label_val):
                path_str = path_bytes.decode("utf-8")
                spec = preprocessor.process_single_file(path_str)
                return spec, np.float32(label_val)

            spec, label = tf.numpy_function(
                _load,
                [path_tensor, label_tensor],
                [tf.float32, tf.float32],
            )
            # Set static shapes so Keras knows dimensions at graph-build time
            spec.set_shape(spec_shape)
            label.set_shape([])
            return spec, label

        ds = tf.data.Dataset.from_tensor_slices(
            (file_paths, np.array(int_labels, dtype=np.float32))
        )

        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(file_paths),
                seed=self.cfg.random_seed,
                reshuffle_each_iteration=True,
            )

        ds = ds.map(_process_file, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.cfg.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _print_class_distribution(labels: List[str], tag: str = "") -> None:
        dist = Counter(labels)
        parts = [f"{cls}: {cnt}" for cls, cnt in sorted(dist.items())]
        print(f"  [{tag}] {' | '.join(parts)}")
