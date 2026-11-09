#!/usr/bin/env python3
"""
evaluate_local_pi.py — Local Raspberry Pi TFLite Model & Preprocessing Evaluator.

Role: Senior MLOps Engineer
Purpose:
  Test the TFLite model directly on the Raspberry Pi using recorded .wav files
  (e.g., in 'collected_background_noise1/' or 'collected_background_noise/').
  
  This test isolates whether real-time False Positives are caused by:
    A) The live microphone stream (sounddevice buffer, DC offset, hardware spikes)
    OR
    B) The TFLite model weights / preprocessing math itself.

Pipeline Consistency:
  Replicates / imports the EXACT preprocessing pipeline from main_system.py
  (SpectrogramProcessor + SystemConfig + TFLiteEngine).
"""

import os
import sys
import glob
import argparse
import numpy as np
import librosa
from scipy import signal

# ─────────────────────────────────────────────────────────────────────────────
# 1. Pipeline Classes (Import from main_system.py or use exact clone fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from main_system import SystemConfig, SpectrogramProcessor, TFLiteEngine
    PIPELINE_SOURCE = "imported directly from main_system.py"
except Exception as e:
    PIPELINE_SOURCE = f"standalone clone (main_system import skipped: {e})"

    from dataclasses import dataclass

    # TFLite Runtime loader supporting all edge and desktop environments
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                import tensorflow as tf
                tflite = tf.lite

    @dataclass
    class SystemConfig:
        """Central configuration identical to main_system.py."""
        sample_rate: int = 16_000
        native_sample_rate: int = 44_100
        duration_sec: float = 7.0
        target_length: int = 112_000  # sample_rate * duration_sec

        # Mel-Spectrogram parameters
        n_mels: int = 64
        n_fft: int = 1024
        hop_length: int = 256
        fmax: int = 4000

        # Model Paths
        stage1_tflite: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "Stage_one_output", "stage1_gatekeeper.tflite")
        cry_threshold: float = 0.5

    class SpectrogramProcessor:
        """Exact replication of main_system.SpectrogramProcessor."""
        def __init__(self, cfg: SystemConfig):
            self.cfg = cfg

        def process(self, native_audio: np.ndarray, native_rate: int) -> np.ndarray:
            # 1. Anti-aliasing resampling if needed
            if native_rate != self.cfg.sample_rate:
                audio = signal.resample_poly(
                    native_audio,
                    up=self.cfg.sample_rate,
                    down=native_rate
                )
            else:
                audio = native_audio

            # 2. Pad or truncate to target_length
            if len(audio) < self.cfg.target_length:
                audio = np.pad(audio, (0, self.cfg.target_length - len(audio)))
            elif len(audio) > self.cfg.target_length:
                audio = audio[:self.cfg.target_length]

            # 3. Log-Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                fmax=self.cfg.fmax,
            )
            # Absolute scaling
            log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)

            # Fixed-range normalization: [-100, 55] dB -> [0, 1]
            log_mel = np.clip((log_mel + 100.0) / 155.0, 0.0, 1.0)

            # Shape: (1, n_mels, time_frames, 1) float32
            return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

    class TFLiteEngine:
        """Exact replication of main_system.TFLiteEngine."""
        def __init__(self, model_path: str, name: str = "model"):
            self.name = name
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"TFLite model not found: {model_path}")

            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        def predict(self, input_data: np.ndarray) -> np.ndarray:
            self.interpreter.set_tensor(
                self.input_details[0]["index"],
                input_data.astype(np.float32),
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
            return output.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main Evaluation Routine
# ─────────────────────────────────────────────────────────────────────────────
def resolve_paths(custom_model=None, custom_noise_dir=None):
    """Auto-detect model and background noise directory on Pi or Dev environment."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Resolve Model Path
    if custom_model and os.path.exists(custom_model):
        model_path = custom_model
    else:
        candidates = [
            os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper.tflite"),
            os.path.join(base_dir, "Stage_one_output", "stage1_model.tflite"),
            os.path.join(base_dir, "stage1_gatekeeper.tflite"),
            os.path.join(base_dir, "stage1_model.tflite"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)

    # 2. Resolve Noise Directory
    if custom_noise_dir and os.path.exists(custom_noise_dir):
        noise_dir = custom_noise_dir
    else:
        dir_candidates = [
            os.path.join(base_dir, "collected_background_noise1"),
            os.path.join(base_dir, "collected_background_noise"),
            os.path.join(os.path.expanduser("~"), "collected_background_noise1"),
            os.path.join(os.path.expanduser("~"), "collected_background_noise"),
        ]
        noise_dir = next((d for d in dir_candidates if os.path.exists(d)), None)

    return model_path, noise_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFLite model on Pi recorded background noise.")
    parser.add_argument("--model", type=str, default=None, help="Path to .tflite model file")
    parser.add_argument("--noise-dir", type=str, default=None, help="Path to noise directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cry decision threshold (default: 0.5)")
    args = parser.parse_args()

    cfg = SystemConfig()
    model_path, noise_dir = resolve_paths(args.model, args.noise_dir)

    print("=" * 68)
    print("  🍼 SMART CRIB — Raspberry Pi Local TFLite Evaluator")
    print("=" * 68)
    print(f"  Pipeline Source : {PIPELINE_SOURCE}")
    print(f"  TFLite Model    : {model_path}")
    print(f"  Noise Directory : {noise_dir}")
    print(f"  Threshold       : {args.threshold:.2f}")
    print("=" * 68)

    if not model_path or not os.path.exists(model_path):
        print(f"\n❌ Error: TFLite model not found! Looked for stage1_gatekeeper.tflite.")
        print("Please check Stage_one_output/ or pass --model <path_to_tflite>")
        sys.exit(1)

    if not noise_dir or not os.path.exists(noise_dir):
        print(f"\n❌ Error: Background noise directory not found!")
        print("Please ensure 'collected_background_noise1' or 'collected_background_noise' exists,")
        print("or pass --noise-dir <path>")
        sys.exit(1)

    # Find all .wav files
    wav_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))
    total_files = len(wav_files)

    if total_files == 0:
        print(f"\n❌ Error: No .wav files found inside '{noise_dir}'")
        sys.exit(1)

    # Initialize Preprocessor and Model
    spec_processor = SpectrogramProcessor(cfg)
    engine = TFLiteEngine(model_path, name="Stage1_Gatekeeper")

    print(f"\n🚀 Evaluating {total_files} background noise samples...")
    print("-" * 68)

    passed_noise = []
    false_positives = []

    for i, wav_path in enumerate(wav_files, start=1):
        filename = os.path.basename(wav_path)

        # 1. Load audio (librosa converts to mono float32 [-1.0, 1.0])
        audio, sr = librosa.load(wav_path, sr=None)

        # 2. Process using the exact SpectrogramProcessor
        spec = spec_processor.process(audio, sr)

        # 3. Predict with TFLite Engine
        output = engine.predict(spec)
        prob = float(output.flatten()[0])

        # 4. Record and display result
        if prob >= args.threshold:
            false_positives.append((filename, prob))
            print(f"  [{i:03d}/{total_files}] ❌ FAILED (CRY)   │ {filename:<24} │ prob = {prob:.4f} 🚨")
        else:
            passed_noise.append((filename, prob))
            print(f"  [{i:03d}/{total_files}] ✅ PASSED (NOISE) │ {filename:<24} │ prob = {prob:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Summary & Diagnostic Report
    # ─────────────────────────────────────────────────────────────────────────
    num_passed = len(passed_noise)
    num_failed = len(false_positives)
    fpr = (num_failed / total_files) * 100.0
    success_rate = (num_passed / total_files) * 100.0

    print("\n" + "=" * 68)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 68)
    print(f"  Total Samples Tested           : {total_files}")
    print(f"  Correctly Identified (NOISE)   : {num_passed:>4} / {total_files} ({success_rate:6.2f}%) ✅")
    print(f"  Misclassified as Cry (FP)      : {num_failed:>4} / {total_files} ({fpr:6.2f}%) ❌")
    print(f"  False Positive Rate (FPR)      : {fpr:6.2f}%")
    print("=" * 68)

    if num_failed > 0:
        print("\n🔍 Misclassified Files Breakdown (False Positives):")
        for fn, p in false_positives:
            print(f"   • {fn:<25} → Cry Prob: {p:.4f}")
        print("-" * 68)

    print("\n💡 MLOps Root Cause Isolation Insight:")
    if fpr <= 5.0:
        print("  ✅ [CASE 1: OFFLINE TFLITE IS CLEAN]")
        print("     The TFLite model and mathematical preprocessing accurately reject noise.")
        print("     Any false positives in real-time 'main_system.py' are almost certainly caused")
        print("     by the LIVE AUDIO STREAM (e.g. microphone buffer underruns, live hardware noise spikes,")
        print("     or sliding window transient accumulation). Check sounddevice and input gain.")
    else:
        print("  ⚠️ [CASE 2: MODEL DOMAIN MISMATCH]")
        print(f"     The TFLite model directly misclassifies {fpr:.1f}% of these static audio files.")
        print("     The model weights themselves are confused by this room/microphone acoustic profile.")
        print("     Solution: Retrain or finetune Stage 1 with these noise files included in NO_RESPONSE.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()

