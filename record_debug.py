import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ── Configuration ──
DURATION_SEC = 7.0
TARGET_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
FMAX = 4000

# Fixed-range normalization constants (matches preprocessor.py)
DB_FLOOR = -80.0  # librosa.power_to_db floor
DB_CEIL = 0.0     # peak-normalized ceiling

def detect_sample_rate() -> int:
    """Detect the default microphone's native sample rate."""
    try:
        dev_info = sd.query_devices(kind="input")
        rate = int(dev_info["default_samplerate"])
        print(f"🎤 Detected Mic: {dev_info['name']} (Native Rate: {rate} Hz)")
        return rate
    except Exception as e:
        print(f"⚠️ Could not detect mic rate, defaulting to 48000 Hz. Error: {e}")
        return 48000

def record_audio(rate: int, duration: float) -> np.ndarray:
    """Record audio using sounddevice."""
    print(f"\n🔴 RECORDING for {duration} seconds... Please speak normally.")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()  # Block until done
    print("✅ Recording complete!")
    return audio[:, 0]

def main():
    hw_rate = detect_sample_rate()
    
    # 1. Record 7 seconds
    native_audio = record_audio(hw_rate, DURATION_SEC)
    
    # Scrub NaNs just in case
    np.nan_to_num(native_audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(native_audio, -1.0, 1.0, out=native_audio)

    # 2. Resample to 16kHz (handles any native rate correctly)
    if hw_rate != TARGET_RATE:
        print(f"🔄 Resampling from {hw_rate} Hz to {TARGET_RATE} Hz...")
        audio_16k = librosa.resample(
            native_audio, orig_sr=hw_rate, target_sr=TARGET_RATE
        )
    else:
        audio_16k = native_audio

    # Ensure exact length (112000 samples)
    target_length = int(DURATION_SEC * TARGET_RATE)
    if len(audio_16k) < target_length:
        audio_16k = np.pad(audio_16k, (0, target_length - len(audio_16k)))
    elif len(audio_16k) > target_length:
        audio_16k = audio_16k[:target_length]

    # Save the Audio to disk
    wav_path = "debug_audio.wav"
    sf.write(wav_path, audio_16k, TARGET_RATE)
    print(f"💾 Saved Audio to: {wav_path}")

    # 3. Preprocess Spectrogram
    print("📊 Generating Spectrogram...")

    mel = librosa.feature.melspectrogram(
        y=audio_16k,
        sr=TARGET_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )
    # Peak-normalize: loudest bin = 0 dB, silence ≈ -80 dB
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Fixed-range normalization: [-80, 0] dB → [0, 1]
    normalized_spec = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)

    # 4. Plot both Spectrograms
    plt.figure(figsize=(12, 8))

    # Plot 1: Raw Log-Mel (Human readable)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(log_mel, sr=TARGET_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmax=FMAX, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Raw Log-Mel Spectrogram (Peak Normalized to 0 dB)')
    plt.tight_layout()

    # Plot 2: Fixed-range normalized (AI Input)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(normalized_spec, sr=TARGET_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmax=FMAX, cmap='viridis')
    plt.colorbar(format='%.2f')
    plt.title('Fixed-Range [0,1] Spectrogram (What the AI Neural Network sees)')
    plt.tight_layout()

    # Save Image
    img_path = "debug_spectrogram.png"
    plt.savefig(img_path)
    print(f"🖼️  Saved Spectrogram Image to: {img_path}")

if __name__ == "__main__":
    main()
