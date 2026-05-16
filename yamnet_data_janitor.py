#!/usr/bin/env python3
"""
yamnet_data_janitor.py — Automated "Data Janitor" for noise dataset cleaning.

Role: Senior Data Engineer
Goal: Filter environmental noise datasets (like ESC-50) to ensure they are 
      completely free of accidental baby cries using Google's YAMNet.
"""

import os
import shutil
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# ── CONFIGURATION ────────────────────────────────────────────────────────────
SOURCE_DIR = "data/noise_esc50"
APPROVED_DIR = "data/approved_NO_RESPONSE"
REJECTED_DIR = "data/rejected_noise"

# YAMNet Class Indices for "Crying"
# 20: Crying, sobbing
# 21: Baby cry, infant cry
CRY_INDICES = [20, 21]
STRICT_THRESHOLD = 0.15  # 15% probability trigger

# ── SETUP ────────────────────────────────────────────────────────────────────
def setup_directories():
    for d in [APPROVED_DIR, REJECTED_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

def load_yamnet():
    print("🧠 Loading YAMNet from TensorFlow Hub...")
    # This might take a moment to download on first run
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

# ── MAIN PROCESSING ──────────────────────────────────────────────────────────
def main():
    setup_directories()
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Source directory '{SOURCE_DIR}' not found.")
        print(f"Please place your raw .wav files in '{SOURCE_DIR}' first.")
        return

    wav_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.wav')]
    if not wav_files:
        print(f"⚠️ No .wav files found in {SOURCE_DIR}")
        return

    model = load_yamnet()
    
    # Track statistics
    stats = {"approved": 0, "rejected": 0, "errors": 0}
    
    print(f"\n🚀 Starting Data Janitor on {len(wav_files)} files...")
    
    for filename in tqdm(wav_files, desc="Cleaning Dataset", unit="file"):
        file_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            # 1. Load and Resample to 16kHz (Required by YAMNet)
            audio, _ = librosa.load(file_path, sr=16000)
            
            # 2. Run YAMNet Inference
            # YAMNet expects a 1D tensor of floats
            scores, embeddings, spectrogram = model(audio)
            
            # scores shape: [num_frames, 521]
            # 3. Extract max probability for cry classes across all frames
            cry_scores = scores.numpy()[:, CRY_INDICES]
            max_cry_prob = np.max(cry_scores)
            
            # 4. Filter Logic
            if max_cry_prob > STRICT_THRESHOLD:
                # REJECTED: Accidental cry detected
                dest_path = os.path.join(REJECTED_DIR, filename)
                shutil.move(file_path, dest_path)
                stats["rejected"] += 1
            else:
                # APPROVED: Clean noise
                dest_path = os.path.join(APPROVED_DIR, filename)
                shutil.move(file_path, dest_path)
                stats["approved"] += 1
                
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")
            stats["errors"] += 1

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("🧹 DATA JANITOR SUMMARY REPORT")
    print("="*50)
    print(f"✅ Approved (Clean Noise) : {stats['approved']}")
    print(f"🚫 Rejected (Cry Detected) : {stats['rejected']}")
    print(f"⚠️ Errors                 : {stats['errors']}")
    print(f"📊 Total Processed        : {len(wav_files)}")
    print("="*50)
    
    if stats['rejected'] > 0:
        print(f"\nTIP: Inspect the files in '{REJECTED_DIR}' to verify YAMNet's findings.")
    print("Done.\n")

if __name__ == "__main__":
    main()
