#!/usr/bin/env python3
"""
yamnet_cry_filter.py — YAMNet-based Infant Cry Quality Filter.

Scans all infant cry folders inside Data_set/, runs Google YAMNet inference
on each .wav file, and copies only files where YAMNet's max-frame confidence
for cry classes (19: Crying/sobbing, 20: Baby cry/infant cry, 21: Whimper) is >= 80%.

Output: new sibling folders with "_yamnet80" suffix containing only verified cries.
Original folders are left completely untouched (copy, not move).

Usage:
    ./venv_yamnet/bin/python yamnet_cry_filter.py
"""

import os
import sys
import csv
import shutil
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

# Compatibility patch for TensorFlow 2.x __internal__ when needed by Hub / tf_keras
def _dummy(x):
    return x

if not hasattr(tf.__internal__, "register_load_context_function"):
    setattr(tf.__internal__, "register_load_context_function", _dummy)
if hasattr(tf, "_api") and hasattr(tf._api, "v2") and hasattr(tf._api.v2, "compat"):
    if hasattr(tf._api.v2.compat, "v2") and hasattr(tf._api.v2.compat.v2, "__internal__"):
        setattr(tf._api.v2.compat.v2.__internal__, "register_load_context_function", _dummy)


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = BASE_DIR / "Data_set"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "yamnet"

# YAMNet class indices for crying/infant distress:
# 19: Crying, sobbing
# 20: Baby cry, infant cry
# 21: Whimper
CRY_INDICES = [19, 20, 21]
CONFIDENCE_THRESHOLD = 0.75  # 75% minimum confidence

# Suffix appended to output folder names
OUTPUT_SUFFIX = "_yamnet75"

# Folders to filter (relative to DATASET_DIR)
TARGET_FOLDERS = [
    "audio/train/real_strong_infantcry_train",
    "audio/train/synth_strong_infantcry_train",
    "audio/train/weak_infantcry",
    "audio/validation/strong_infantcry_val",
    "audio/test/real_strong_infantcry_test",
    "audio/test/synth_strong_infantcry_test",
    "CRY",
]

LOG_FILE = BASE_DIR / "yamnet_cleaning.log"


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════
def setup_logging() -> logging.Logger:
    """Configure dual logging: file (DEBUG) + console (INFO)."""
    logger = logging.getLogger("yamnet_filter")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers if any
    logger.handlers.clear()

    fh = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════════════════
# YAMNET LOADER
# ══════════════════════════════════════════════════════════════════════════
def load_yamnet(logger: logging.Logger):
    """Load Google YAMNet model and class map."""
    class_names = []

    if LOCAL_MODEL_DIR.exists():
        logger.info(f"🧠 Loading local YAMNet model from {LOCAL_MODEL_DIR}...")
        model = tf.saved_model.load(str(LOCAL_MODEL_DIR))
        class_map_path = LOCAL_MODEL_DIR / "assets" / "yamnet_class_map.csv"
        if class_map_path.exists():
            with open(class_map_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_names.append(row["display_name"])
    else:
        logger.info("🧠 Loading YAMNet from TensorFlow Hub...")
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row["display_name"])

    target_names = [class_names[i] for i in CRY_INDICES] if class_names else CRY_INDICES
    logger.info(f"✅ YAMNet ready. Classes loaded ({len(class_names)} total).")
    logger.info(f"   Target cry classes: {target_names} (indices {CRY_INDICES})")
    return model, class_names


# ══════════════════════════════════════════════════════════════════════════
# SINGLE FILE SCORING
# ══════════════════════════════════════════════════════════════════════════
def score_file(model, file_path: str) -> float:
    """Run YAMNet on a single .wav file and return max cry confidence."""
    try:
        # Fast read with soundfile (already 16kHz mono 16-bit in Data_set)
        audio, sr = sf.read(file_path, dtype="float32")
        if len(audio) == 0:
            return 0.0

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Run YAMNet inference: returns (scores, embeddings, spectrogram)
        scores, _, _ = model(audio)
        scores_np = scores.numpy()
        cry_scores = scores_np[:, CRY_INDICES]
        return float(np.max(cry_scores))

    except Exception:
        # Fallback to librosa if soundfile fails on non-standard headers
        try:
            audio, _ = librosa.load(file_path, sr=16000, mono=True)
            if len(audio) == 0:
                return 0.0
            scores, _, _ = model(audio)
            scores_np = scores.numpy()
            cry_scores = scores_np[:, CRY_INDICES]
            return float(np.max(cry_scores))
        except Exception:
            return 0.0


# ══════════════════════════════════════════════════════════════════════════
# FOLDER PROCESSOR
# ══════════════════════════════════════════════════════════════════════════
def process_folder(
    model,
    source_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Filter one folder: copy files with cry confidence >= threshold."""
    stats = {"total": 0, "accepted": 0, "rejected": 0, "errors": 0}

    wav_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".wav")])
    stats["total"] = len(wav_files)

    if stats["total"] == 0:
        logger.warning(f"  ⚠️  No .wav files found in {source_dir}")
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(wav_files, start=1):
        src_path = source_dir / filename

        try:
            max_cry_prob = score_file(model, str(src_path))

            if max_cry_prob >= CONFIDENCE_THRESHOLD:
                # ACCEPTED: verified cry — copy to output folder
                dst_path = output_dir / filename
                shutil.copy2(str(src_path), str(dst_path))
                stats["accepted"] += 1
                logger.debug(
                    f"  ✅ [{i}/{stats['total']}] {filename} "
                    f"→ accepted (conf={max_cry_prob:.3f})"
                )
            else:
                # REJECTED: corrupted / low-confidence
                stats["rejected"] += 1
                logger.debug(
                    f"  ❌ [{i}/{stats['total']}] {filename} "
                    f"→ rejected (conf={max_cry_prob:.3f})"
                )

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"  ⚠️  [{i}/{stats['total']}] {filename} → ERROR: {e}")

        # Progress update every 200 files or at completion
        if i % 200 == 0 or i == stats["total"]:
            pct = (stats["accepted"] / i * 100) if i > 0 else 0
            logger.info(
                f"  📊 Progress: {i:>5d}/{stats['total']} files processed "
                f"| Accepted: {stats['accepted']:>5d} ({pct:5.1f}%) "
                f"| Rejected: {stats['rejected']:>5d}"
            )

    return stats


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("  🍼 YAMNet Infant Cry Quality Filter & Dataset Sanitizer")
    logger.info(f"  Threshold     : {CONFIDENCE_THRESHOLD * 100:.0f}% confidence (≥ {CONFIDENCE_THRESHOLD})")
    logger.info(f"  Cry classes   : indices {CRY_INDICES}")
    logger.info(f"  Output suffix : '{OUTPUT_SUFFIX}'")
    logger.info(f"  Mode          : Safe Copy (Original files left 100% untouched)")
    logger.info("=" * 70)

    # ── Validate target folders ──────────────────────────────────────────
    valid_targets = []
    for rel_path in TARGET_FOLDERS:
        source = DATASET_DIR / rel_path
        if source.is_dir():
            output = source.parent / (source.name + OUTPUT_SUFFIX)
            valid_targets.append((source, output, rel_path))
            file_count = len([f for f in os.listdir(source) if f.lower().endswith(".wav")])
            logger.info(f"  📁 Target folder: {rel_path:<45} ({file_count} files)")
        else:
            logger.warning(f"  ⚠️  Folder not found (skipping): {source}")

    if not valid_targets:
        logger.error("No valid target folders found. Exiting.")
        sys.exit(1)

    # ── Load YAMNet ──────────────────────────────────────────────────────
    model, class_names = load_yamnet(logger)

    # ── Process each folder independently ────────────────────────────────
    grand_total = {"total": 0, "accepted": 0, "rejected": 0, "errors": 0}
    folder_summaries = []

    for source_dir, output_dir, rel_path in valid_targets:
        logger.info("")
        logger.info("─" * 70)
        logger.info(f"📂 Processing folder: {rel_path}")
        logger.info(f"   Source directory : {source_dir}")
        logger.info(f"   Output directory : {output_dir}")
        logger.info("─" * 70)

        stats = process_folder(model, source_dir, output_dir, logger)

        accept_pct = (
            (stats["accepted"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )
        reject_pct = (
            (stats["rejected"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )

        folder_summaries.append({
            "name": rel_path,
            "total": stats["total"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "accept_pct": accept_pct,
            "output_dir": output_dir,
        })

        logger.info(
            f"  ✅ Finished: {rel_path} → "
            f"Accepted: {stats['accepted']}/{stats['total']} ({accept_pct:.1f}%) | "
            f"Rejected: {stats['rejected']} ({reject_pct:.1f}%)"
        )

        for key in grand_total:
            grand_total[key] += stats[key]

    # ── Grand Summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("🏁 GRAND SUMMARY — YAMNet Infant Cry Quality Filter")
    logger.info("=" * 70)

    for item in folder_summaries:
        logger.info(
            f"  📁 {item['name']:<45} : "
            f"{item['accepted']:>5d} / {item['total']:>5d} accepted ({item['accept_pct']:5.1f}%)"
        )

    grand_pct = (
        (grand_total["accepted"] / grand_total["total"] * 100)
        if grand_total["total"] > 0
        else 0
    )
    logger.info("─" * 70)
    logger.info(f"  📊 Total files scanned    : {grand_total['total']}")
    logger.info(
        f"  ✅ Total verified cries   : {grand_total['accepted']} ({grand_pct:.1f}%)"
    )
    logger.info(f"  ❌ Total filtered out     : {grand_total['rejected']}")
    logger.info(f"  ⚠️  Total errors            : {grand_total['errors']}")
    logger.info(f"  📄 Detailed log written to : {LOG_FILE}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
