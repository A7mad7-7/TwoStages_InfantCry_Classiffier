#!/usr/bin/env python3
"""
import_urbansound8k.py — Import selected UrbanSound8K classes into
the Stage 1 NO_RESPONSE training set.

Reads the UrbanSound8K.csv metadata, selects 200 random samples from
each of the specified classes, processes them (resample to 16 kHz,
pad/truncate to 7 s), and saves them to clean_data/Stage_One_Data/NO_RESPONSE/.

Usage:
    python import_urbansound8k.py
"""

import os
import random
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────
URBANSOUND_DIR = Path("UrbanSound8K")
CSV_PATH = URBANSOUND_DIR / "UrbanSound8K.csv"

OUTPUT_DIR = Path("clean_data/Stage_One_Data/NO_RESPONSE")

TARGET_SR = 16_000
TARGET_DURATION = 7.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 112,000

SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42

# Classes to import (classID → class name)
SELECTED_CLASSES = {
    0: "air_conditioner",
    2: "children_playing",
    3: "dog_bark",
    8: "siren",
}

# ── Logging ──────────────────────────────────────────────────────────────
def setup_logging():
    logger = logging.getLogger("import_urban")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ── Audio Processing (mirrors prepare_raw_data.py logic) ─────────────────
def process_and_save(src_path: Path, out_path: Path, logger: logging.Logger) -> bool:
    """Load audio, resample to 16 kHz mono, pad/truncate to 7 s, save as WAV."""
    try:
        y, _ = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        logger.warning(f"Failed to load {src_path.name}: {e}")
        return False

    if len(y) == 0:
        logger.warning(f"Empty audio: {src_path.name}")
        return False

    # Pad or truncate to exactly 7 seconds
    if len(y) >= TARGET_SAMPLES:
        # Take only the first 7 seconds
        y = y[:TARGET_SAMPLES]
    else:
        # Pad with silence
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")

    sf.write(str(out_path), y, TARGET_SR, subtype="PCM_16")
    return True


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    logger = setup_logging()
    random.seed(RANDOM_SEED)

    logger.info("=" * 60)
    logger.info("UrbanSound8K → NO_RESPONSE Importer")
    logger.info("=" * 60)

    # 1. Validate paths
    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Read metadata
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded metadata: {len(df)} total entries")

    # 3. Filter and sample
    selected_files = []

    for class_id, class_name in SELECTED_CLASSES.items():
        class_df = df[df["classID"] == class_id]
        available = len(class_df)

        if available < SAMPLES_PER_CLASS:
            logger.warning(
                f"Class {class_id} ({class_name}): only {available} available, "
                f"using all of them."
            )
            sampled = class_df
        else:
            sampled = class_df.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)

        logger.info(
            f"  Class {class_id:2d} ({class_name:20s}): "
            f"{len(sampled)}/{available} selected"
        )

        for _, row in sampled.iterrows():
            fold = f"fold{row['fold']}"
            filename = row["slice_file_name"]
            src_path = URBANSOUND_DIR / fold / filename
            selected_files.append((src_path, class_name, filename))

    logger.info(f"\nTotal files to import: {len(selected_files)}")

    # 4. Process and save
    success = 0
    skipped = 0

    for src_path, class_name, filename in tqdm(selected_files, desc="Importing", unit="file"):
        if not src_path.exists():
            logger.warning(f"File not found: {src_path}")
            skipped += 1
            continue

        # Prefix with class name to avoid filename collisions
        out_name = f"urban_{class_name}_{filename}"
        out_path = OUTPUT_DIR / out_name

        if out_path.exists():
            logger.debug(f"Already exists, skipping: {out_name}")
            success += 1
            continue

        if process_and_save(src_path, out_path, logger):
            success += 1
        else:
            skipped += 1

    # 5. Summary
    total_no_response = len(list(OUTPUT_DIR.glob("*.wav")))

    logger.info("\n" + "=" * 60)
    logger.info("IMPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ✅ Successfully imported : {success}")
    logger.info(f"  ⚠️  Skipped              : {skipped}")
    logger.info(f"  📊 Total NO_RESPONSE now : {total_no_response} files")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
