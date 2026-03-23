#!/usr/bin/env python3
"""
main_system.py — Smart Crib "Brain" for Raspberry Pi 4.

Production-grade, multi-threaded State Machine that orchestrates:
  • Sliding-window audio capture (7 s @ 16 kHz, updated every 2 s)
  • Two-stage cascaded TFLite inference (Cry Detection → Cry Classification)
  • Arduino serial I/O (sensor reads + motor commands)
  • Lightweight Flask dashboard for parent-facing status

Architecture (4 threads):
  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐
  │ Audio Thread  │  │ Serial I/O Thread│  │  Flask Thread   │
  │ (sounddevice) │  │  (pyserial)      │  │  (dashboard)    │
  └──────┬───────┘  └──────┬───────────┘  └────────┬───────┘
         │                 │                        │
         ▼                 ▼                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Main Thread — AI State Machine             │
  │   LISTENING ──▶ CRY_DETECTED ──▶ COOLDOWN ──▶ ...      │
  └─────────────────────────────────────────────────────────┘

Author : Senior IoT & Edge AI Engineering Team
Target : Raspberry Pi 4 (ARMv8, 4 GB RAM)
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
import time
import enum
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np

# Audio
try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[WARN] sounddevice not installed. Audio capture disabled.")

# Spectrogram
import librosa

# TFLite Runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback: full TensorFlow (development machine)
    import tensorflow as tf
    tflite = tf.lite

# Serial
try:
    import serial
except ImportError:
    serial = None
    print("[WARN] pyserial not installed. Arduino serial disabled.")

# Web Dashboard
from flask import Flask, jsonify, render_template_string


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SystemConfig:
    """Central configuration for the Smart Crib system."""

    # ── Audio ──────────────────────────────────────────────────────────
    sample_rate: int = 16_000
    duration_sec: float = 7.0
    target_length: int = 112_000        # sample_rate × duration_sec
    slide_interval_sec: float = 2.0     # How often to run inference

    # ── Mel-Spectrogram (MUST match training) ─────────────────────────
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    fmax: int = 4000

    # ── TFLite Model Paths ─────────────────────────────────────────────
    stage1_tflite: str = os.path.join(BASE_DIR, "Stage_one_output", "stage1_model.tflite")
    stage2_tflite: str = os.path.join(BASE_DIR, "Stage_two_output", "stage2_expert.tflite")

    # ── State Machine ──────────────────────────────────────────────────
    cry_threshold: float = 0.5         # Stage 1 sigmoid threshold
    cooldown_sec: float = 60.0         # Seconds to sleep after cry detected

    # ── Arduino Serial ─────────────────────────────────────────────────
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    serial_timeout: float = 1.0

    # ── Flask Dashboard ────────────────────────────────────────────────
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 5000

    # ── Class Labels ───────────────────────────────────────────────────
    stage1_labels = ["NO_RESPONSE", "CRY"]
    stage2_labels = ["Pain", "Hungry", "Tired"]


# ═══════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════
class CribState(enum.Enum):
    """State Machine states for the Smart Crib."""
    LISTENING     = "Listening"
    CRY_DETECTED  = "Crying"
    COOLDOWN      = "Soothing"


# ═══════════════════════════════════════════════════════════════════════
# SHARED STATE (Thread-safe)
# ═══════════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe shared data store accessed by all threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = CribState.LISTENING
        self._cry_reason: Optional[str] = None
        self._cry_confidence: float = 0.0
        self._sensor_data: Dict[str, Any] = {
            "heart_rate": None,
            "temperature": None,
        }
        self._last_update = datetime.now().isoformat()

    # ── Crib State ─────────────────────────────────────────────────────
    @property
    def state(self) -> CribState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: CribState):
        with self._lock:
            self._state = value
            self._last_update = datetime.now().isoformat()

    # ── Cry Info ───────────────────────────────────────────────────────
    def set_cry_info(self, reason: str, confidence: float):
        with self._lock:
            self._cry_reason = reason
            self._cry_confidence = confidence
            self._last_update = datetime.now().isoformat()

    def clear_cry_info(self):
        with self._lock:
            self._cry_reason = None
            self._cry_confidence = 0.0

    # ── Sensors ────────────────────────────────────────────────────────
    def update_sensors(self, heart_rate=None, temperature=None):
        with self._lock:
            if heart_rate is not None:
                self._sensor_data["heart_rate"] = heart_rate
            if temperature is not None:
                self._sensor_data["temperature"] = temperature

    # ── JSON snapshot for the dashboard ────────────────────────────────
    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self._state.value,
                "cry_reason": self._cry_reason,
                "cry_confidence": round(self._cry_confidence, 3),
                "sensors": dict(self._sensor_data),
                "last_update": self._last_update,
            }


# ═══════════════════════════════════════════════════════════════════════
# AUDIO BUFFER (Thread 1)
# ═══════════════════════════════════════════════════════════════════════
class AudioBuffer:
    """Continuously captures audio via sounddevice and maintains a
    7-second rolling buffer at 16 kHz (mono, float32).

    The buffer is updated continuously via the sounddevice callback.
    The main AI loop reads the latest 7-second snapshot every 2 seconds.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        # Pre-allocate a ring buffer of exactly target_length samples
        self._buffer = np.zeros(cfg.target_length, dtype=np.float32)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self.logger = logging.getLogger("AudioBuffer")

    def start(self):
        """Start the audio input stream."""
        if sd is None:
            self.logger.error("sounddevice not available. Cannot capture audio.")
            return

        self._running = True
        try:
            self._stream = sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.cfg.sample_rate * 0.1),  # 100 ms blocks
                callback=self._audio_callback,
            )
            self._stream.start()
            self.logger.info("🎙️  Audio stream started (16 kHz, mono).")
        except Exception as e:
            self.logger.error(f"Failed to start audio stream: {e}")
            self._running = False

    def stop(self):
        """Stop and close the audio stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self.logger.info("Audio stream stopped.")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — appends new audio to the rolling buffer."""
        if status:
            self.logger.warning(f"Audio status: {status}")
        # indata shape: (frames, 1) → flatten to 1-D
        new_samples = indata[:, 0]
        with self._lock:
            # Shift buffer left, append new samples at the end
            n = len(new_samples)
            self._buffer = np.roll(self._buffer, -n)
            self._buffer[-n:] = new_samples

    def get_snapshot(self) -> np.ndarray:
        """Return a copy of the current 7-second audio buffer."""
        with self._lock:
            return self._buffer.copy()


# ═══════════════════════════════════════════════════════════════════════
# SPECTROGRAM PROCESSOR (Replicates AudioPreprocessor logic)
# ═══════════════════════════════════════════════════════════════════════
class SpectrogramProcessor:
    """Lightweight spectrogram extractor for real-time inference.
    Mirrors the exact pipeline used during training:
      raw audio → Log-Mel Spectrogram (fmax=4000) → Z-score → channel dim

    NOTE: For production, the Z-score mean/std should be saved during
    training and loaded here. We use reasonable defaults as fallback.
    """

    def __init__(self, cfg: SystemConfig, mean: float = -30.0, std: float = 15.0):
        self.cfg = cfg
        # These should ideally be loaded from training artifacts
        self.mean = mean
        self.std = std
        self.logger = logging.getLogger("SpectrogramProcessor")

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Convert a 7-second audio buffer to a normalized spectrogram.

        Parameters
        ----------
        audio : np.ndarray, shape (112000,), float32

        Returns
        -------
        spec : np.ndarray, shape (1, 64, time_frames, 1), float32
               Ready for TFLite inference (batch dim included).
        """
        # Ensure correct length
        if len(audio) < self.cfg.target_length:
            audio = np.pad(audio, (0, self.cfg.target_length - len(audio)))
        elif len(audio) > self.cfg.target_length:
            audio = audio[:self.cfg.target_length]

        # Log-Mel Spectrogram (identical to training preprocessor)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmax=self.cfg.fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Z-score normalize
        log_mel = (log_mel - self.mean) / self.std

        # Add batch + channel dimensions: (1, n_mels, time, 1)
        return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# TFLITE INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════
class TFLiteEngine:
    """Loads and runs inference on a quantized TFLite model."""

    def __init__(self, model_path: str, name: str = "model"):
        self.name = name
        self.logger = logging.getLogger(f"TFLite.{name}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        in_shape = self.input_details[0]["shape"]
        in_dtype = self.input_details[0]["dtype"]
        self.logger.info(
            f"Loaded {name}: input={in_shape} ({in_dtype.__name__}), "
            f"output={self.output_details[0]['shape']}"
        )

        # Quantization parameters (for INT8 models)
        self._input_quant = self.input_details[0].get("quantization_parameters", {})
        self._output_quant = self.output_details[0].get("quantization_parameters", {})

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run a single forward pass.

        Handles INT8 quantization transparently:
          float32 input → quantize → invoke → dequantize → float32 output

        Parameters
        ----------
        input_data : np.ndarray, float32, shape matching model input

        Returns
        -------
        output : np.ndarray, float32
        """
        input_dtype = self.input_details[0]["dtype"]

        # Quantize input if model expects INT8
        if input_dtype == np.int8:
            scale = self._input_quant.get("scales", [1.0])[0]
            zero_point = self._input_quant.get("zero_points", [0])[0]
            input_data = (input_data / scale + zero_point).astype(np.int8)
        elif input_dtype == np.uint8:
            scale = self._input_quant.get("scales", [1.0])[0]
            zero_point = self._input_quant.get("zero_points", [0])[0]
            input_data = (input_data / scale + zero_point).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Dequantize output if INT8
        output_dtype = self.output_details[0]["dtype"]
        if output_dtype in (np.int8, np.uint8):
            scale = self._output_quant.get("scales", [1.0])[0]
            zero_point = self._output_quant.get("zero_points", [0])[0]
            output = (output.astype(np.float32) - zero_point) * scale

        return output.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# ARDUINO SERIAL MANAGER (Thread 2)
# ═══════════════════════════════════════════════════════════════════════
class ArduinoManager:
    """Background thread that continuously reads sensor data from Arduino
    and sends motor commands ('C' = Cry/ON, 'S' = Stop/OFF).

    Expected Arduino CSV format (incoming):
      BPM,Temp\n    e.g.  72,36.50\n
    Baud: 115200
    """

    def __init__(self, cfg: SystemConfig, shared: SharedState):
        self.cfg = cfg
        self.shared = shared
        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.logger = logging.getLogger("Arduino")

    def start(self):
        """Open serial port and start the reader thread."""
        if serial is None:
            self.logger.warning("pyserial not installed. Arduino disabled.")
            return

        try:
            self._ser = serial.Serial(
                port=self.cfg.serial_port,
                baudrate=self.cfg.serial_baud,
                timeout=self.cfg.serial_timeout,
            )
            time.sleep(2)  # Wait for Arduino to reset after serial open
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, name="ArduinoReader", daemon=True
            )
            self._thread.start()
            self.logger.info(
                f"🔌 Arduino connected on {self.cfg.serial_port} "
                f"@ {self.cfg.serial_baud} baud."
            )
        except Exception as e:
            self.logger.error(f"Failed to open serial port: {e}")

    def stop(self):
        """Stop the reader thread and close serial."""
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.logger.info("Serial port closed.")

    def send_command(self, cmd: str):
        """Send a single-character command to the Arduino.

        'C' → Cry detected / Motor ON
        'S' → Stop / Motor OFF
        """
        if self._ser is None or not self._ser.is_open:
            self.logger.warning(f"Cannot send '{cmd}': serial not connected.")
            return
        try:
            self._ser.write(cmd.encode("ascii"))
            self._ser.flush()
            self.logger.info(f"📤 Sent command '{cmd}' to Arduino.")
        except Exception as e:
            self.logger.error(f"Serial write error: {e}")

    def _read_loop(self):
        """Continuously read CSV sensor data from Arduino.
        Expected format: 'BPM,Temp\n' e.g. '72,36.50\n'
        """
        while self._running:
            try:
                if self._ser is None or not self._ser.is_open:
                    time.sleep(1)
                    continue

                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or "," not in line:
                    continue  # Skip empty lines or non-CSV (e.g. "CRIB_READY")

                parts = line.split(",")
                if len(parts) == 2:
                    bpm  = int(parts[0])
                    temp = float(parts[1])
                    self.shared.update_sensors(
                        heart_rate=bpm,
                        temperature=temp,
                    )
            except (ValueError, IndexError):
                pass  # Malformed CSV line — skip silently
            except Exception as e:
                self.logger.error(f"Serial read error: {e}")
                time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════
# FLASK WEB DASHBOARD (Thread 3)
# ═══════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Crib Dashboard</title>
    <style>
        :root {
            --bg: #0f172a; --card-bg: #1e293b; --accent: #38bdf8;
            --text: #e2e8f0; --text-dim: #94a3b8;
            --green: #22c55e; --yellow: #eab308; --red: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background: var(--bg); color: var(--text);
            min-height: 100vh; padding: 2rem;
        }
        h1 { text-align: center; font-size: 1.8rem; margin-bottom: 1.5rem; }
        h1 span { color: var(--accent); }
        .grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.2rem; max-width: 900px; margin: 0 auto;
        }
        .card {
            background: var(--card-bg); border-radius: 12px;
            padding: 1.5rem; border: 1px solid #334155;
        }
        .card h2 {
            font-size: 0.85rem; text-transform: uppercase;
            color: var(--text-dim); letter-spacing: 0.05em; margin-bottom: 0.8rem;
        }
        .status-value {
            font-size: 2rem; font-weight: 700;
        }
        .status-listening  { color: var(--green);  }
        .status-crying     { color: var(--red);    }
        .status-soothing   { color: var(--yellow); }
        .sensor-row {
            display: flex; justify-content: space-between;
            padding: 0.6rem 0; border-bottom: 1px solid #334155;
        }
        .sensor-row:last-child { border-bottom: none; }
        .sensor-label { color: var(--text-dim); }
        .sensor-val   { font-weight: 600; font-size: 1.1rem; }
        .footer {
            text-align: center; color: var(--text-dim);
            margin-top: 2rem; font-size: 0.8rem;
        }
    </style>
    <script>
        async function refresh() {
            try {
                const res = await fetch('/api/status');
                const d = await res.json();

                const el = document.getElementById('status');
                el.textContent = d.status;
                el.className = 'status-value status-' + d.status.toLowerCase();

                document.getElementById('reason').textContent =
                    d.cry_reason ? d.cry_reason + ' (' + d.cry_confidence + ')' : '—';

                document.getElementById('hr').textContent =
                    d.sensors.heart_rate != null ? d.sensors.heart_rate + ' bpm' : '—';
                document.getElementById('temp').textContent =
                    d.sensors.temperature != null ? d.sensors.temperature + ' °C' : '—';

                document.getElementById('updated').textContent =
                    'Last update: ' + d.last_update;
            } catch(e) { console.error(e); }
        }
        setInterval(refresh, 2000);
        window.onload = refresh;
    </script>
</head>
<body>
    <h1>🍼 <span>Smart Crib</span> Dashboard</h1>
    <div class="grid">
        <div class="card">
            <h2>Crib Status</h2>
            <div id="status" class="status-value status-listening">Listening</div>
        </div>
        <div class="card">
            <h2>Cry Reason</h2>
            <div id="reason" class="status-value" style="font-size:1.4rem;">—</div>
        </div>
        <div class="card">
            <h2>Vital Signs</h2>
            <div class="sensor-row">
                <span class="sensor-label">❤️ Heart Rate</span>
                <span class="sensor-val" id="hr">—</span>
            </div>
            <div class="sensor-row">
                <span class="sensor-label">🌡️ Temperature</span>
                <span class="sensor-val" id="temp">—</span>
            </div>
        </div>
    </div>
    <p class="footer" id="updated">Last update: —</p>
</body>
</html>
"""


def create_dashboard_app(shared: SharedState) -> Flask:
    """Create the Flask application with JSON API and HTML dashboard."""
    app = Flask(__name__)
    # Silence Flask request logs in production
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/status")
    def api_status():
        return jsonify(shared.to_dict())

    return app


# ═══════════════════════════════════════════════════════════════════════
# MAIN AI STATE MACHINE (Main Thread / Thread 4)
# ═══════════════════════════════════════════════════════════════════════
class SmartCribBrain:
    """
    The main orchestrator. Runs in the main thread.

    State Transitions:
      LISTENING ─[cry detected]──▶ CRY_DETECTED
      CRY_DETECTED ──────────────▶ COOLDOWN
      COOLDOWN ──[still crying]──▶ COOLDOWN  (loop)
      COOLDOWN ──[stopped crying]─▶ LISTENING
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("Brain")
        self.shared = SharedState()
        self._shutdown = threading.Event()

        # ── Sub-systems ────────────────────────────────────────────────
        self.audio = AudioBuffer(cfg)
        self.spec_processor = SpectrogramProcessor(cfg)
        self.arduino = ArduinoManager(cfg, self.shared)

        # ── Load TFLite Models ─────────────────────────────────────────
        self.logger.info("Loading TFLite models …")
        self.stage1 = TFLiteEngine(cfg.stage1_tflite, name="Stage1_Gatekeeper")
        self.stage2 = TFLiteEngine(cfg.stage2_tflite, name="Stage2_Expert")
        self.logger.info("✅ Both TFLite models loaded.")

    # ── Inference helpers ──────────────────────────────────────────────
    def _run_stage1(self, spec: np.ndarray) -> float:
        """Return the CRY probability (0.0 – 1.0) from Stage 1."""
        output = self.stage1.predict(spec)
        # Stage 1: Dense(1, sigmoid) → shape (1, 1)
        return float(output.flatten()[0])

    def _run_stage2(self, spec: np.ndarray) -> tuple:
        """Return (class_name, confidence) from Stage 2.
        Stage 2 outputs logits → softmax → argmax.
        """
        output = self.stage2.predict(spec)  # shape (1, 3)
        logits = output.flatten()
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        idx = int(np.argmax(probs))
        return self.cfg.stage2_labels[idx], float(probs[idx])

    # ── Main Loop ──────────────────────────────────────────────────────
    def run(self):
        """Start all sub-systems and enter the State Machine loop."""

        # Start sub-systems
        self.audio.start()
        self.arduino.start()

        # Start Flask in a daemon thread
        app = create_dashboard_app(self.shared)
        flask_thread = threading.Thread(
            target=lambda: app.run(
                host=self.cfg.dashboard_host,
                port=self.cfg.dashboard_port,
                debug=False,
                use_reloader=False,
            ),
            name="FlaskDashboard",
            daemon=True,
        )
        flask_thread.start()
        self.logger.info(
            f"🌐 Dashboard running at http://{self.cfg.dashboard_host}:"
            f"{self.cfg.dashboard_port}/"
        )

        self.logger.info("🧠 State Machine started — entering LISTENING state.")

        try:
            while not self._shutdown.is_set():
                current_state = self.shared.state

                if current_state == CribState.LISTENING:
                    self._handle_listening()

                elif current_state == CribState.CRY_DETECTED:
                    self._handle_cry_detected()

                elif current_state == CribState.COOLDOWN:
                    self._handle_cooldown()

        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown signal received.")
        finally:
            self._cleanup()

    # ── State Handlers ─────────────────────────────────────────────────
    def _handle_listening(self):
        """STATE_LISTENING: Grab audio snapshot → Stage 1 inference.
        If CRY detected → transition to CRY_DETECTED.
        Otherwise sleep for slide_interval and repeat.
        """
        audio = self.audio.get_snapshot()
        spec = self.spec_processor.process(audio)

        cry_prob = self._run_stage1(spec)
        self.logger.debug(f"Stage 1 → CRY probability: {cry_prob:.3f}")

        if cry_prob >= self.cfg.cry_threshold:
            self.logger.info(
                f"🚨 CRY DETECTED (prob={cry_prob:.3f}). "
                f"Transitioning to CRY_DETECTED."
            )
            self.shared.state = CribState.CRY_DETECTED
        else:
            # No cry — wait and listen again
            self._shutdown.wait(timeout=self.cfg.slide_interval_sec)

    def _handle_cry_detected(self):
        """STATE_CRY_DETECTED:
        1. Run Stage 2 to classify the cry reason.
        2. Send 'C' to Arduino (start motor).
        3. Update dashboard.
        4. Transition immediately to COOLDOWN.
        """
        # Re-grab latest audio for Stage 2
        audio = self.audio.get_snapshot()
        spec = self.spec_processor.process(audio)

        reason, confidence = self._run_stage2(spec)
        self.logger.info(f"🔍 Cry Reason: {reason} (confidence={confidence:.3f})")

        # Update shared state
        self.shared.set_cry_info(reason, confidence)

        # Send motor command to Arduino
        self.arduino.send_command("C")

        # Transition to COOLDOWN
        self.shared.state = CribState.COOLDOWN
        self.logger.info(
            f"💤 Entering COOLDOWN for {self.cfg.cooldown_sec}s …"
        )

    def _handle_cooldown(self):
        """STATE_COOLDOWN:
        Sleep for cooldown_sec, then check if baby is still crying.
        - Still crying → stay in COOLDOWN.
        - Stopped → send 'S' to Arduino → return to LISTENING.
        """
        # Non-blocking sleep — allows shutdown event to interrupt
        self._shutdown.wait(timeout=self.cfg.cooldown_sec)

        if self._shutdown.is_set():
            return

        # Wake up and check with Stage 1
        audio = self.audio.get_snapshot()
        spec = self.spec_processor.process(audio)
        cry_prob = self._run_stage1(spec)

        if cry_prob >= self.cfg.cry_threshold:
            self.logger.info(
                f"😢 Still crying after cooldown (prob={cry_prob:.3f}). "
                f"Staying in COOLDOWN."
            )
            # Stay in COOLDOWN — the loop will call _handle_cooldown again
        else:
            self.logger.info(
                f"😊 Baby calmed down (prob={cry_prob:.3f}). "
                f"Sending STOP to Arduino. Returning to LISTENING."
            )
            self.arduino.send_command("S")
            self.shared.clear_cry_info()
            self.shared.state = CribState.LISTENING

    # ── Cleanup ────────────────────────────────────────────────────────
    def _cleanup(self):
        """Graceful shutdown of all sub-systems."""
        self._shutdown.set()
        self.audio.stop()
        self.arduino.send_command("S")  # Safety: stop motor
        self.arduino.stop()
        self.logger.info("✅ All systems shut down cleanly.")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def main():
    """Configure logging and launch the Smart Crib Brain."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = SystemConfig()

    print("=" * 64)
    print("  🍼  SMART CRIB — Cascaded Infant Cry Classifier")
    print("  Target   : Raspberry Pi 4")
    print(f"  Stage 1  : {cfg.stage1_tflite}")
    print(f"  Stage 2  : {cfg.stage2_tflite}")
    print(f"  Serial   : {cfg.serial_port} @ {cfg.serial_baud}")
    print(f"  Dashboard: http://{cfg.dashboard_host}:{cfg.dashboard_port}/")
    print("=" * 64)

    brain = SmartCribBrain(cfg)
    brain.run()


if __name__ == "__main__":
    main()
