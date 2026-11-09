"""
data_loader.py — File discovery, leakage-free group splitting, and NumPy array
creation for Stage 1 (Gatekeeper).

Golden Rule: Zero Data Leakage across train/val/test splits:
  1. CRY: Uses pre-split directories from Data_set/audio/ (train, validation, test)
     filtered via YAMNet quality verification.
  2. NO_RESPONSE: Entire video sessions / podcasts are assigned as indivisible units
     to either train, val, or test. Independent original_samples are split proportionally.
"""

import os
import glob
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class DataLoader:
    """Handles dataset discovery, leakage-free group splitting, and NumPy array creation."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : Config-like object with attributes:
            cry_dataset_dir, no_response_dataset_dir, yamnet_filter_suffix,
            train_video_groups, val_video_groups, test_video_groups,
            class_names, train_ratio, val_ratio, test_ratio, random_seed
        """
        self.cfg = config
        np.random.seed(config.random_seed)
        self._cached_splits: Optional[Dict[str, Tuple[List[str], List[str]]]] = None

    # ── Leakage-Free Dataset Splitting (Primary Entry Point) ───────────
    def load_splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Load stage-one train, validation, and test splits with ZERO data leakage.

        Returns
        -------
        dict with keys 'train', 'val', 'test', each mapping to
        (file_paths_list, labels_list).
        """
        if self._cached_splits is not None:
            return self._cached_splits

        # ── 1. CRY Class from Data_set/audio/{train, validation, test} ──
        cry_splits = self._load_cry_splits()

        # ── 2. NO_RESPONSE Class with Video Group Splitting ─────────────
        no_resp_splits = self._load_no_response_splits()

        # ── 3. Combine and Deterministically Shuffle ────────────────────
        splits: Dict[str, Tuple[List[str], List[str]]] = {}
        for split_key in ["train", "val", "test"]:
            c_paths = cry_splits[split_key]
            nr_paths = no_resp_splits[split_key]

            combined_paths = c_paths + nr_paths
            combined_labels = ["CRY"] * len(c_paths) + ["NO_RESPONSE"] * len(nr_paths)

            # Deterministic shuffle
            rng = np.random.RandomState(self.cfg.random_seed + (0 if split_key == "train" else 1))
            indices = np.arange(len(combined_paths))
            rng.shuffle(indices)

            shuffled_paths = [combined_paths[i] for i in indices]
            shuffled_labels = [combined_labels[i] for i in indices]

            splits[split_key] = (shuffled_paths, shuffled_labels)

        print("\n" + "=" * 65)
        print("  [DataLoader] Zero-Leakage Dataset Partition Summary")
        print("=" * 65)
        for name in ["train", "val", "test"]:
            paths, lbls = splits[name]
            print(f"  {name.upper():>10s} set: {len(paths):5d} total samples")
            self._print_class_distribution(lbls, tag=f"{name.upper()}")
        print("=" * 65 + "\n")

        self._cached_splits = splits
        return splits

    # ── Helper: Load CRY splits from Data_set/audio ────────────────────
    def _load_cry_splits(self) -> Dict[str, List[str]]:
        cry_splits = {"train": [], "val": [], "test": []}
        split_dir_map = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }

        cry_base = getattr(self.cfg, "cry_dataset_dir", None)
        if not cry_base or not os.path.isdir(cry_base):
            # Fallback if Data_set/audio does not exist
            return cry_splits

        suffix = getattr(self.cfg, "yamnet_filter_suffix", "_yamnet75")

        for split_key, sub_dir in split_dir_map.items():
            folder_path = os.path.join(cry_base, sub_dir)
            if not os.path.isdir(folder_path):
                continue

            # Look for verified yamnet folders first
            pattern = os.path.join(folder_path, f"*infantcry*{suffix}")
            matched_dirs = sorted(glob.glob(pattern))

            # Fallback to any infantcry folder if suffix not matched
            if not matched_dirs:
                matched_dirs = sorted(glob.glob(os.path.join(folder_path, "*infantcry*")))

            for d in matched_dirs:
                wav_files = sorted(glob.glob(os.path.join(d, "*.wav")))
                cry_splits[split_key].extend(wav_files)

        return cry_splits

    # ── Helper: Load NO_RESPONSE with Group-Based Splitting ────────────
    def _load_no_response_splits(self) -> Dict[str, List[str]]:
        no_resp_dir = getattr(self.cfg, "no_response_dataset_dir", None)
        if not no_resp_dir or not os.path.isdir(no_resp_dir):
            return {"train": [], "val": [], "test": []}

        train_groups = set(getattr(self.cfg, "train_video_groups", []))
        val_groups = set(getattr(self.cfg, "val_video_groups", []))
        test_groups = set(getattr(self.cfg, "test_video_groups", []))

        def collect_files_from_groups(groups: set) -> List[str]:
            files: List[str] = []
            for g in sorted(groups):
                group_dir = os.path.join(no_resp_dir, g)
                if os.path.isdir(group_dir):
                    wavs = sorted([
                        os.path.join(group_dir, f)
                        for f in os.listdir(group_dir)
                        if f.lower().endswith(".wav")
                    ])
                    files.extend(wavs)
            return files

        nr_train = collect_files_from_groups(train_groups)
        nr_val = collect_files_from_groups(val_groups)
        nr_test = collect_files_from_groups(test_groups)

        # Check for unassigned folders
        all_subdirs = {
            d for d in os.listdir(no_resp_dir)
            if os.path.isdir(os.path.join(no_resp_dir, d))
        }
        assigned_groups = train_groups | val_groups | test_groups | {"original_samples"}
        unassigned = all_subdirs - assigned_groups
        if unassigned:
            print(f"[DataLoader] Notice: adding unassigned NO_RESPONSE folders to train: {unassigned}")
            nr_train.extend(collect_files_from_groups(unassigned))

        # Split independent original_samples
        orig_dir = os.path.join(no_resp_dir, "original_samples")
        if os.path.isdir(orig_dir):
            orig_files = sorted([
                os.path.join(orig_dir, f)
                for f in os.listdir(orig_dir)
                if f.lower().endswith(".wav")
            ])
            if orig_files:
                o_train, o_temp = train_test_split(
                    orig_files,
                    test_size=(1.0 - self.cfg.train_ratio),
                    random_state=self.cfg.random_seed,
                )
                val_frac = self.cfg.val_ratio / (self.cfg.val_ratio + self.cfg.test_ratio)
                o_val, o_test = train_test_split(
                    o_temp,
                    test_size=(1.0 - val_frac),
                    random_state=self.cfg.random_seed,
                )
                nr_train.extend(o_train)
                nr_val.extend(o_val)
                nr_test.extend(o_test)

        return {"train": nr_train, "val": nr_val, "test": nr_test}

    # ── Backward Compatibility: load_file_paths & split_data ──────────
    def load_file_paths(self) -> Tuple[List[str], List[str]]:
        """Walk dataset directories or call load_splits() if structured."""
        if hasattr(self.cfg, "cry_dataset_dir") and os.path.isdir(self.cfg.cry_dataset_dir):
            splits = self.load_splits()
            all_paths = splits["train"][0] + splits["val"][0] + splits["test"][0]
            all_labels = splits["train"][1] + splits["val"][1] + splits["test"][1]
            return all_paths, all_labels

        # Fallback to simple directory walking
        file_paths: List[str] = []
        labels: List[str] = []
        for class_name in sorted(self.cfg.class_names):
            class_dir = os.path.join(self.cfg.dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(".wav"):
                        file_paths.append(os.path.join(root, fname))
                        labels.append(class_name)
        return file_paths, labels

    def split_data(
        self,
        file_paths: List[str],
        labels: List[str],
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Return precomputed leakage-free splits if available, else stratified split."""
        if self._cached_splits is not None:
            return self._cached_splits

        if hasattr(self.cfg, "cry_dataset_dir") and os.path.isdir(self.cfg.cry_dataset_dir):
            return self.load_splits()

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
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

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
        preprocessor : AudioPreprocessor

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
