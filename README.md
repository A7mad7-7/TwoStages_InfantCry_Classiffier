# 🍼 Smart Crib: Two-Stage Edge AI Infant Cry Classifier

> **Status:** Software Architecture Complete. V1 Models Trained and Quantized. Pending physical deployment and testing on Raspberry Pi 4.

## 📌 Executive Summary
This project implements a complete, end-to-end Edge AI system for a "Smart Crib". It transforms raw microphone audio and physical sensors (Heart Rate, SpO2, Temperature) into an automated soothing mechanism. The core innovation is a **Two-Stage Cascaded Neural Network** designed to run locally on a Raspberry Pi, minimizing false positives and optimizing battery/CPU usage.

---

## 🏗 System Architecture & Engineering Decisions

### 1. The AI Pipeline (Why Two Stages & Custom Transfer Learning?)
A major challenge in baby monitors is false alarms (e.g., adult speech or TV triggering the crib). Additionally, we faced a severe data scarcity issue: we had only ~300 reliable, verified samples per specific cry type, surrounded by highly corrupted and noisy audio data. 

To solve this without relying on generic, heavy pre-trained audio models (which lack domain specialization and bloat the Edge device), I designed a custom Two-Stage Cascaded Pipeline:

*   **Stage 1: The Gatekeeper (Binary CNN):** An always-on, lightweight CNN trained on a massive dataset of mixed audio and injected noise. It achieves **98%+ accuracy** in distinguishing `CRY` from `NO_RESPONSE` (ambient noise, speech, silence), strictly preventing the motor from triggering false alarms.
*   **Stage 2: The Expert (Custom Transfer Learning):** Only executes if Stage 1 detects a cry. To overcome the lack of labeled multi-class data, I used **Transfer Learning** directly from Stage 1. By freezing the robust convolutional base of the Gatekeeper (which had already mastered extracting core acoustic cry features) and attaching a new classification head, we significantly boosted the multi-class accuracy (`Hungry`, `Pain`, `Tired`). This approach maintained full architectural control and kept the model footprint extremely tiny for Edge deployment.

*Note: The repository includes an `xgboost_pipeline` directory. This was our baseline classical ML experiment. While it performed exceptionally well at binary detection (F1: 0.94), it hit an acoustic ceiling for multi-class cry separation, validating our pivot to the deep learning cascaded architecture.*

### 2. Audio Preprocessing (The Domain Engineering)
Audio processing on Edge devices requires strict standardization to prevent "Shortcut Learning" by the neural network:
*   **Fixed Window:** All audio is strictly padded/truncated to a 7.0-second sliding window.
*   **Microphone-Agnostic Feature Extraction:** We extract Log-Mel Spectrograms at 16kHz. **Crucial Fix:** We explicitly set `fmax=4000` in the STFT conversion. This crops high-frequency dead zones, forcing the CNN to learn the actual formant shapes of the cry rather than overfitting to the microphone quality or sampling rate artifacts.

### 3. Hardware Sensor Fusion (Raspberry Pi + Arduino)
To ensure real-time performance without blocking the AI inference thread, the hardware logic is distributed:
*   **The Brain (Raspberry Pi 4):** Runs the TFLite models, manages the state machine, and hosts a local lightweight Web Dashboard (Flask/FastAPI) for parents.
*   **The Spinal Cord (Arduino):** Connects via Serial USB. It continuously polls the I2C/Analog sensors (DS18B20, Pulse Sensor) and drives the NEMA 23 Stepper Motor using non-blocking timers (`millis()`).
*   **The Handshake:** When the Pi detects a cry, it sends a single-byte command (`C`) to the Arduino to start a 3-minute motor rocking cooldown.

### 4. Edge Optimization
Both CNN models undergo **INT8 Post-Training Quantization (PTQ)**. The spatial dimensions are aggressively downsampled through 4 Convolutional blocks before flattening, bringing the parameter count from ~7.2M down to ~500k per model. The final `.tflite` artifacts are under 1MB each.

---

## 🚀 How to Run (Local Testing)
1. Install requirements: `pip install -r requirements.txt`
2. Run the data cleaner: `python prepare_raw_data.py`
3. Train Stage 1: `python stage_one_pipeline.py`
4. Train Stage 2: `python stage_two_pipeline.py`
5. Test the main system loop (Mock Sensors): `python main_system.py`
