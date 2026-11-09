#!/usr/bin/env python3
"""
evaluate_real_world.py — Evaluate model accuracy on real-world background noise recorded from Raspberry Pi.

Goal:
Test the model's ability to recognize background noise from the 'collected_background_noise' folder:
- Successful samples: Correctly classified as NOISE (cry probability < 0.5 / True Negative).
- Failed samples: Misclassified as CRY (cry probability >= 0.5 / False Positive).
"""

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from preprocessor import AudioPreprocessor
from stage_one_config import Config


def load_model_helper(model_path):
    """Load the model in either Keras (.keras) or TFLite (.tflite) format."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")

    if model_path.endswith(".tflite"):
        print(f"📦 Loading TFLite model: {os.path.basename(model_path)}...")
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                from tensorflow import lite as tflite

        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        def predict_fn(features):
            interpreter.set_tensor(input_details[0]["index"], features.astype(np.float32))
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            return float(out.flatten()[0])

        return predict_fn
    else:
        print(f"🧠 Loading Keras model: {os.path.basename(model_path)}...")
        model = tf.keras.models.load_model(model_path)

        def predict_fn(features):
            return float(model.predict(features, verbose=0)[0][0])

        return predict_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on real-world noise samples (collected_background_noise)")
    parser.add_argument("--model", type=str, default=None, help="Path to model file (.keras or .tflite)")
    parser.add_argument("--noise-dir", type=str, default=None, help="Path to noise directory (default: collected_background_noise)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for cry classification (default: 0.5)")
    args = parser.parse_args()

    cfg = Config()
    preprocessor = AudioPreprocessor(cfg)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Automatically determine model path
    if args.model:
        model_path = args.model
    else:
        default_keras = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper.keras")
        default_tflite = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper.tflite")
        default_finetuned = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper_finetuned.keras")

        if os.path.exists(default_keras):
            model_path = default_keras
        elif os.path.exists(default_tflite):
            model_path = default_tflite
        elif os.path.exists(default_finetuned):
            model_path = default_finetuned
        else:
            model_path = default_keras

    # 2. Set noise directory path
    # noise_dir = args.noise_dir or os.path.join(base_dir, "collected_background_noise1")
    noise_dir = args.noise_dir or os.path.join(base_dir, "collected_background_noise")


    if not os.path.exists(noise_dir):
        print(f"❌ Error: Directory not found: {noise_dir}")
        print("Please ensure the 'collected_background_noise' folder exists and contains audio files.")
        return

    wav_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))
    total_samples = len(wav_files)

    if total_samples == 0:
        print(f"❌ Error: No .wav files found inside directory: {noise_dir}")
        return

    # 3. Load model
    try:
        predict_fn = load_model_helper(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"\n🚀 Evaluating {total_samples} noise samples from '{os.path.basename(noise_dir)}'...")
    print(f"🎚️ Decision Threshold: {args.threshold:.2f}\n")

    successful_noise = []  # Succeeded: correctly classified as NOISE (prob < threshold)
    failed_noise = []      # Failed: misclassified as CRY (prob >= threshold, False Positive)

    for i, wav_file in enumerate(wav_files, start=1):
        filename = os.path.basename(wav_file)
        features = preprocessor.process_single_file(wav_file)
        features = np.expand_dims(features, axis=0)
        
        prob = predict_fn(features)

        if prob >= args.threshold:
            # Failed: noise sound misclassified as cry (False Positive)
            failed_noise.append((filename, prob))
            print(f"[{i:03d}/{total_samples}] ❌ FAILED | {filename:<25} -> prob={prob:.3f}  (Classified as CRY 🚨)")
        else:
            # Succeeded: noise sound correctly identified as NOISE (True Negative)
            successful_noise.append((filename, prob))
            print(f"[{i:03d}/{total_samples}] ✅ PASSED | {filename:<25} -> prob={prob:.3f}  (Classified as NOISE)")

    # 4. Report & Statistics
    num_success = len(successful_noise)
    num_failed = len(failed_noise)
    success_rate = (num_success / total_samples) * 100
    error_rate = (num_failed / total_samples) * 100

    print("\n" + "=" * 60)
    print("🎯 Real-World Background Noise Evaluation Report")
    print("=" * 60)
    print(f"• Evaluated Model            : {os.path.basename(model_path)}")
    print(f"• Evaluated Directory        : {noise_dir}")
    print(f"• Decision Threshold         : {args.threshold:.2f}")
    print("-" * 60)
    print(f"• Total Samples Tested       : {total_samples}")
    print(f"• Successfully Identified (NOISE) : {num_success:>4} / {total_samples}  ({success_rate:6.2f}%) ✅")
    print(f"• Misclassified / False Alarms (CRY) : {num_failed:>4} / {total_samples}  ({error_rate:6.2f}%) ❌")
    print("-" * 60)
    print(f"📈 Success Rate (True Negative Rate) : {success_rate:6.2f}%")
    print(f"📉 Error Rate (False Positive Rate)  : {error_rate:6.2f}%")
    print("=" * 60)

    if num_failed > 0:
        print("\n🔍 Details of Misclassified Samples (False Positives):")
        for fn, p in failed_noise:
            print(f"   - {fn:<25} -> Cry Probability: {p:.3f}")
        print("=" * 60)

    if error_rate <= 5.0:
        print("🌟 Excellent performance: False positive rate is very low and suitable for production!")
    elif error_rate <= 15.0:
        print("⚠️ Moderate performance: Some false alarms observed; tuning threshold or fine-tuning may help.")
    else:
        print("🚨 Poor performance: High false alarm rate (> 15%); model needs retraining or sensitivity adjustment.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
