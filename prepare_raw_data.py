#!/usr/bin/env python3
"""
Phase 0 — Raw Data Standardization for Stage 1 Binary Classifier.

Reads raw audio files (wav, ogg, mp4, webm) from:
    Stage_One_Data/CRY/
    Stage_One_Data/NO-Response/

Outputs clean, standardized 7-second mono 16 kHz .wav files into:
    clean_data/Stage_One_Data/CRY/
    clean_data/Stage_One_Data/NO_RESPONSE/

Processing per file:
  1. Load → mono → resample to 16 kHz
  2. Files > 7 s  → slice into non-overlapping 7 s segments (discard remainder)
  3. Files < 7 s  → zero-pad to exactly 7 s
  4. Files == 7 s → kept as-is
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────
TARGET_SR = 16_000          # 16 kHz
TARGET_DURATION = 7.0       # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 112 000

RAW_BASE = Path("Stage_One_Data")
CLEAN_BASE = Path("clean_data/Stage_One_Data")

# Map raw subfolder names → clean subfolder names
CLASS_MAP = {
    "CRY": "CRY",
    "NO-Response": "NO_RESPONSE",
}

SUPPORTED_EXTENSIONS = {".wav", ".ogg", ".mp4", ".webm", ".mp3", ".flac", ".m4a"}

LOG_FILE = "prepare_raw_data.log"


# ──────────────────────────── Helpers ──────────────────────────────────
def setup_logging() -> logging.Logger:
    """Configure dual logging: file (DEBUG) + console (INFO)."""
    logger = logging.getLogger("prepare_raw_data")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_audio(filepath: Path, sr: int, logger: logging.Logger) -> np.ndarray | None:
    """
    Load an audio file as mono at the requested sample rate.

    For video containers (.mp4, .webm) that librosa/soundfile cannot
    decode natively, fall back to ffmpeg to extract audio first.
    """
    ext = filepath.suffix.lower()

    # ── 1. Try native librosa load (handles wav, ogg, flac, mp3, etc.) ──
    if ext not in {".mp4", ".webm"}:
        try:
            y, _ = librosa.load(str(filepath), sr=sr, mono=True)
            return y
        except Exception as e:
            logger.warning(f"librosa failed on {filepath.name}: {e}")
            # Fall through to ffmpeg as a last resort

    # ── 2. ffmpeg fallback for video containers / stubborn files ─────────
    try:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        cmd = [
            "ffmpeg", "-y", "-i", str(filepath),
            "-vn",                       # discard video
            "-ac", "1",                  # mono
            "-ar", str(sr),              # resample
            "-sample_fmt", "s16",        # 16-bit PCM
            tmp_wav.name,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        y, _ = librosa.load(tmp_wav.name, sr=sr, mono=True)
        return y
    except Exception as e:
        logger.error(f"ffmpeg also failed on {filepath.name}: {e}")
        return None
    finally:
        if os.path.exists(tmp_wav.name):
            os.remove(tmp_wav.name)


def process_file(
    y: np.ndarray,
    src_stem: str,
    out_dir: Path,
    logger: logging.Logger,
) -> int:
    """
    Chunk / pad a mono 16 kHz signal into exactly 7-second segments
    and write them to *out_dir*.  Returns the number of segments saved.
    """
    n_samples = len(y)
    saved = 0

    if n_samples >= TARGET_SAMPLES:
        # ── Chunk into non-overlapping 7s windows ──
        n_chunks = n_samples // TARGET_SAMPLES
        for i in range(n_chunks):
            start = i * TARGET_SAMPLES
            segment = y[start : start + TARGET_SAMPLES]
            out_name = f"{src_stem}_part{i + 1:04d}.wav" if n_chunks > 1 else f"{src_stem}.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), segment, TARGET_SR, subtype="PCM_16")
            saved += 1
        remainder = n_samples - n_chunks * TARGET_SAMPLES
        if remainder > 0:
            logger.debug(
                f"Discarded {remainder} trailing samples "
                f"({remainder / TARGET_SR:.2f}s) from {src_stem}"
            )
    else:
        # ── Pad short files with silence ──
        pad_length = TARGET_SAMPLES - n_samples
        segment = np.pad(y, (0, pad_length), mode="constant")
        out_path = out_dir / f"{src_stem}.wav"
        sf.write(str(out_path), segment, TARGET_SR, subtype="PCM_16")
        logger.debug(
            f"Padded {src_stem} with {pad_length} samples "
            f"({pad_length / TARGET_SR:.2f}s of silence)"
        )
        saved += 1

    return saved


# ──────────────────────────── Main ─────────────────────────────────────
def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Phase 0 — Raw Data Standardization (Stage 1)")
    logger.info("=" * 60)

    # ── Validate raw directories ────────────────────────────────────────
    for raw_sub in CLASS_MAP:
        raw_path = RAW_BASE / raw_sub
        if not raw_path.is_dir():
            logger.error(f"Raw data directory not found: {raw_path}")
            sys.exit(1)

    # ── Create clean output directories ─────────────────────────────────
    for clean_sub in CLASS_MAP.values():
        (CLEAN_BASE / clean_sub).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {CLEAN_BASE / clean_sub}")

    # ── Collect all files ───────────────────────────────────────────────
    file_list: list[tuple[Path, Path]] = []  # (src_file, dest_dir)
    for raw_sub, clean_sub in CLASS_MAP.items():
        raw_dir = RAW_BASE / raw_sub
        out_dir = CLEAN_BASE / clean_sub
        for f in sorted(raw_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_list.append((f, out_dir))

    logger.info(f"Found {len(file_list)} audio files to process.")

    # ── Process ─────────────────────────────────────────────────────────
    total_segments = 0
    skipped = 0

    for src_file, out_dir in tqdm(file_list, desc="Processing", unit="file"):
        try:
            y = load_audio(src_file, TARGET_SR, logger)
            if y is None or len(y) == 0:
                logger.warning(f"SKIPPED (empty/unreadable): {src_file}")
                skipped += 1
                continue

            n = process_file(y, src_file.stem, out_dir, logger)
            total_segments += n

        except Exception as e:
            logger.error(f"SKIPPED (unexpected error): {src_file} — {e}")
            skipped += 1
            continue

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"  Files processed : {len(file_list) - skipped}")
    logger.info(f"  Files skipped   : {skipped}")
    logger.info(f"  Total 7s segments saved: {total_segments}")

    for clean_sub in CLASS_MAP.values():
        count = len(list((CLEAN_BASE / clean_sub).glob("*.wav")))
        logger.info(f"  {clean_sub:15s}: {count} files")

    logger.info(f"Output written to: {CLEAN_BASE.resolve()}")
    logger.info(f"Full log saved to: {LOG_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
