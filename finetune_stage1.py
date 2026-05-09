import os
import glob
import random
import numpy as np
import librosa
import tensorflow as tf
from preprocessor import AudioPreprocessor
from stage_one_config import Config
from tqdm import tqdm

def create_balanced_dataset(cfg, bg_noise_dir, clean_data_dir):
    """
    Load 200 NO_RESPONSE samples from Pi.
    Load 200 CRY samples randomly from clean_data.
    Returns (X, y) ready for fine-tuning.
    """
    preprocessor = AudioPreprocessor(cfg)

    # 1. Load Pi Background Noise (NO_RESPONSE = 0)
    bg_files = glob.glob(os.path.join(bg_noise_dir, "*.wav"))
    if len(bg_files) == 0:
        raise ValueError(f"No background noise found in {bg_noise_dir}")
    
    # Optional: limit to 200 just in case
    bg_files = bg_files[:200]
    
    # 2. Load Original CRY Data (CRY = 1)
    cry_dir = os.path.join(clean_data_dir, "CRY")
    if not os.path.exists(cry_dir):
        raise ValueError(f"Original CRY directory not found at {cry_dir}")
        
    all_cry_files = glob.glob(os.path.join(cry_dir, "*.wav"))
    if len(all_cry_files) < len(bg_files):
        raise ValueError("Not enough CRY files to balance the dataset!")
        
    # Randomly select matching number of CRY files to perfectly balance
    selected_cry_files = random.sample(all_cry_files, len(bg_files))

    print(f"Dataset Balancing:")
    print(f" - {len(bg_files)} NO_RESPONSE files (from Raspberry Pi)")
    print(f" - {len(selected_cry_files)} CRY files (from Original Dataset)")
    
    # 3. Extract Features
    all_files = bg_files + selected_cry_files
    labels = [0] * len(bg_files) + [1] * len(selected_cry_files)
    
    # Simple extraction loop
    print("Extracting features...")
    specs = []
    for f in tqdm(all_files):
        specs.append(preprocessor.process_single_file(f))
    X = np.array(specs)
    y = np.array(labels)
    
    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    return X[indices], y[indices]

def main():
    cfg = Config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    bg_noise_dir = os.path.join(base_dir, "collected_background_noise")
    clean_data_dir = os.path.join(base_dir, "clean_data", "Stage_One_Data")
    model_path = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper.keras")
    finetuned_model_path = os.path.join(base_dir, "Stage_one_output", "stage1_gatekeeper_finetuned.keras")

    if not os.path.exists(model_path):
        print(f"❌ Error: Original model not found at {model_path}")
        return

    # 1. Create Fine-tuning Dataset
    print("\n🛠️ Preparing Fine-Tuning Dataset...")
    try:
        X_train, y_train = create_balanced_dataset(cfg, bg_noise_dir, clean_data_dir)
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return

    # 2. Load the Model
    print("\n🧠 Loading Original Gatekeeper Model...")
    model = tf.keras.models.load_model(model_path)

    # 3. Compile with a standard finetuning Learning Rate
    print("⚙️ Compiling with Learning Rate (Adam 5e-4)...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Fine-Tune
    print("\n🚀 Starting Fine-Tuning (20 Epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,          # Small batch size for fine-tuning
        validation_split=0.2    # 20% validation to monitor over-fitting
    )

    # 5. Save Model
    model.save(finetuned_model_path)
    print(f"\n✅ Fine-tuning complete! Saved to: {finetuned_model_path}")
    print("\nYou should now run `requantize_models.py` to generate the new .tflite model.")

if __name__ == "__main__":
    main()
