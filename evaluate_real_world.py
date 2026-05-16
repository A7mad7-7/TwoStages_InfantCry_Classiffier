import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from preprocessor import AudioPreprocessor
from stage_one_config import Config

def main():
    cfg = Config()
    preprocessor = AudioPreprocessor(cfg)

    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper_finetuned.keras")
    model_path = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper.keras")


    bg_noise_dir = os.path.join(base_dir, "collected_background_noise")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    if not os.path.exists(bg_noise_dir):
        print(f"❌ Error: Background noise directory not found at {bg_noise_dir}")
        print("Please copy the folder from your Raspberry Pi to this directory.")
        return

    wav_files = glob.glob(os.path.join(bg_noise_dir, "*.wav"))
    if len(wav_files) == 0:
        print(f"❌ Error: No .wav files found in {bg_noise_dir}")
        return

    # 2. Load model
    print("Loading Gatekeeper model...")
    model = tf.keras.models.load_model(model_path)

    print(f"\n🚀 Evaluating {len(wav_files)} real-world background noise samples...")
    
    false_positives = 0
    total_samples = len(wav_files)
    
    for wav_file in wav_files:
        # Apply standard AudioPreprocessor
        features = preprocessor.process_single_file(wav_file)
        
        # Add batch dimension: shape becomes (1, 64, 438, 1)
        features = np.expand_dims(features, axis=0)
        
        # Predict
        prob = model.predict(features, verbose=0)[0][0]
        
        if prob >= 0.5:
            false_positives += 1
            print(f"🚨 FALSE POSITIVE: {os.path.basename(wav_file)} -> prob={prob:.3f}")

    # Results
    fpr = (false_positives / total_samples) * 100
    print("\n" + "="*50)
    print("🎯 REAL-WORLD EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples Tested : {total_samples}")
    print(f"False Positives (CRY): {false_positives}")
    print(f"False Positive Rate  : {fpr:.2f}%")
    
    if fpr > 5.0:
        print("\n⚠️ The model is hallucinating on this microphone's audio!")
        print("We must run `finetune_stage1.py` to fix this out-of-distribution error.")
    else:
        print("\n✅ The model performed well on the real-world audio!")

if __name__ == "__main__":
    main()
