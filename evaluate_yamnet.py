import os
import glob
import csv
from collections import Counter

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

def load_class_names(class_map_csv):
    """Loads the YAMNet class mapping from the CSV file."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bg_noise_dir = os.path.join(base_dir, "collected_background_noise")

    if not os.path.exists(bg_noise_dir):
        print(f"❌ Error: Directory not found -> {bg_noise_dir}")
        return

    wav_files = glob.glob(os.path.join(bg_noise_dir, "*.wav"))
    if len(wav_files) == 0:
        print(f"❌ Error: No .wav files found in {bg_noise_dir}")
        return

    # 1. Load YAMNet Model from TensorFlow Hub
    print("🌍 Downloading/Loading YAMNet from TensorFlow Hub...")
    try:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    except Exception as e:
        print(f"❌ Failed to load YAMNet. Make sure you have 'tensorflow_hub' installed.")
        print(f"Try running: pip install tensorflow_hub")
        return

    # Load the 521 class names
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = load_class_names(class_map_path)

    print(f"\n🚀 Evaluating {len(wav_files)} background noise samples with YAMNet...")
    
    top_predictions = []
    
    for wav_file in wav_files:
        # Load audio at 16kHz (YAMNet requirement)
        y, sr = librosa.load(wav_file, sr=16000)
        
        # Run Inference
        # YAMNet returns:
        # scores: (N_frames, 521)
        # embeddings: (N_frames, 1024)
        # spectrogram: (N_frames, 64)
        scores, embeddings, spectrogram = yamnet_model(y)
        
        # Average the scores across all time frames in the 7-second clip
        mean_scores = np.mean(scores, axis=0)
        
        # Get the top predicted class index
        top_class_idx = np.argmax(mean_scores)
        top_class_name = class_names[top_class_idx]
        
        top_predictions.append(top_class_name)

    # Summarize results
    print("\n" + "="*50)
    print("🎯 YAMNet DIAGNOSIS OF YOUR BACKGROUND NOISE")
    print("="*50)
    
    counter = Counter(top_predictions)
    for class_name, count in counter.most_common(10):
        percentage = (count / len(wav_files)) * 100
        print(f" {class_name.ljust(25)} : {count:3d} files ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
