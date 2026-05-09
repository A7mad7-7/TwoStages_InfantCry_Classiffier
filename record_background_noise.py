import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

# ── Configuration ──
DURATION_SEC = 7.0
TARGET_RATE = 16000
NUM_SAMPLES = 200
OUTPUT_DIR = "collected_background_noise"

def detect_sample_rate() -> int:
    try:
        dev_info = sd.query_devices(kind="input")
        rate = int(dev_info["default_samplerate"])
        print(f"🎤 Detected Mic: {dev_info['name']} (Native Rate: {rate} Hz)")
        return rate
    except Exception as e:
        print(f"⚠️ Could not detect mic rate, defaulting to 48000 Hz. Error: {e}")
        return 48000

def record_audio(rate: int, duration: float) -> np.ndarray:
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hw_rate = detect_sample_rate()
    
    print(f"\n🚀 Starting Data Collection: {NUM_SAMPLES} samples of {DURATION_SEC}s each.")
    print("Please go about your normal activities. This will take about 23 minutes.")
    time.sleep(3)

    for i in range(1, NUM_SAMPLES + 1):
        print(f"\n🔴 Recording sample {i}/{NUM_SAMPLES}...")
        
        native_audio = record_audio(hw_rate, DURATION_SEC)
        
        # Scrub NaNs
        np.nan_to_num(native_audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(native_audio, -1.0, 1.0, out=native_audio)

        # Resample to 16kHz (handles any native rate correctly)
        if hw_rate != TARGET_RATE:
            audio_16k = librosa.resample(
                native_audio, orig_sr=hw_rate, target_sr=TARGET_RATE
            )
        else:
            audio_16k = native_audio

        # Ensure exact length
        target_length = int(DURATION_SEC * TARGET_RATE)
        if len(audio_16k) < target_length:
            audio_16k = np.pad(audio_16k, (0, target_length - len(audio_16k)))
        elif len(audio_16k) > target_length:
            audio_16k = audio_16k[:target_length]

        # Save to disk
        wav_path = os.path.join(OUTPUT_DIR, f"background_{i:03d}.wav")
        sf.write(wav_path, audio_16k, TARGET_RATE)
        print(f"💾 Saved: {wav_path}")

    print("\n✅ All 200 background noise samples collected successfully!")

if __name__ == "__main__":
    main()
