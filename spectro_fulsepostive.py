import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from preprocessor import AudioPreprocessor

class Config:
    """Configuration mapping the exact parameters used in training."""
    sample_rate: int = 16_000
    duration_sec: float = 7.0
    target_length: int = 112_000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    fmax: int = 4000

# List of all misclassified background noise samples
false_positives = [
    "background_002.wav", "background_003.wav", "background_004.wav", "background_005.wav",
    "background_006.wav", "background_007.wav", "background_008.wav", "background_009.wav",
    "background_012.wav", "background_013.wav", "background_014.wav", "background_015.wav",
    "background_019.wav", "background_020.wav", "background_021.wav", "background_022.wav",
    "background_024.wav", "background_026.wav", "background_027.wav", "background_028.wav",
    "background_029.wav", "background_030.wav", "background_031.wav", "background_032.wav",
    "background_033.wav", "background_034.wav", "background_035.wav", "background_036.wav",
    "background_037.wav", "background_038.wav", "background_039.wav", "background_040.wav",
    "background_041.wav", "background_042.wav", "background_063.wav", "background_064.wav",
    "background_067.wav", "background_070.wav", "background_072.wav", "background_075.wav",
    "background_081.wav", "background_084.wav", "background_085.wav", "background_086.wav",
    "background_087.wav", "background_088.wav", "background_089.wav", "background_090.wav",
    "background_091.wav", "background_092.wav", "background_093.wav", "background_094.wav",
    "background_095.wav", "background_096.wav", "background_097.wav", "background_098.wav",
    "background_099.wav", "background_100.wav", "background_101.wav", "background_107.wav",
    "background_109.wav", "background_112.wav", "background_113.wav", "background_122.wav",
    "background_123.wav", "background_126.wav", "background_127.wav", "background_128.wav",
    "background_129.wav", "background_130.wav", "background_152.wav", "background_159.wav",
    "background_160.wav", "background_161.wav", "background_162.wav", "background_163.wav",
    "background_164.wav", "background_165.wav", "background_166.wav", "background_167.wav",
    "background_168.wav", "background_169.wav", "background_170.wav", "background_171.wav",
    "background_172.wav", "background_173.wav", "background_174.wav", "background_175.wav",
    "background_176.wav", "background_177.wav", "background_178.wav", "background_179.wav",
    "background_180.wav", "background_181.wav", "background_182.wav", "background_183.wav",
    "background_184.wav", "background_185.wav", "background_186.wav", "background_188.wav",
    "background_189.wav", "background_190.wav", "background_199.wav", "background_200.wav"
]

def visualize_false_positives(audio_dir: str, save_dir: str = None):
    """
    Process and visualize the misclassified audio files.
    If save_dir is provided, images will be saved instead of shown interactively.
    """
    cfg = Config()
    preprocessor = AudioPreprocessor(cfg)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for fname in false_positives:
        file_path = os.path.join(audio_dir, fname)
        
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue
            
        # Extract processed spectrogram: shape (n_mels, time_frames, 1)
        spec = preprocessor.process_single_file(file_path)
        
        # Squeeze the channel dimension for visualization: shape (n_mels, time_frames)
        spec_2d = spec.squeeze(axis=-1)
        
        plt.figure(figsize=(10, 4))
        
        # Display the spectrogram
        # Data is already normalized to [0, 1] range in preprocessor
        img = librosa.display.specshow(
            spec_2d, 
            sr=cfg.sample_rate, 
            hop_length=cfg.hop_length, 
            x_axis='time', 
            y_axis='mel', 
            fmax=cfg.fmax, 
            cmap='viridis',
            vmin=0.0,
            vmax=1.0
        )
        
        plt.colorbar(img, format='%.2f', label='Normalized Amplitude')
        plt.title(f"False Positive Spectrogram: {fname}")
        plt.tight_layout()
        
        if save_dir:
            out_path = os.path.join(save_dir, f"{fname.replace('.wav', '.png')}")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved: {out_path}")
        else:
            plt.show()

if __name__ == "__main__":
    # Define the directory containing the background noise .wav files
    AUDIO_DIRECTORY = "./collected_background_noise1"
    
    # Optional: Define output directory to save images automatically
    SAVE_DIRECTORY = "./false_positives_plots" 
    
    visualize_false_positives(AUDIO_DIRECTORY, SAVE_DIRECTORY)