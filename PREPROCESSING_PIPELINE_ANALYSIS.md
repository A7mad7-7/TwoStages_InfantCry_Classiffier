# Comprehensive Preprocessing Pipeline Analysis: Two-Stage Edge AI Infant Cry Classifier

---

## 1. Executive Overview & Architectural Philosophy

In embedded audio classification systems—particularly medical-grade and smart nursery IoT applications—the **audio preprocessing pipeline is not merely a data conversion step; it is the primary deterministic inductive bias of the entire artificial intelligence architecture**. 

In the **Two-Stage Cascaded Infant Cry Classifier** running on an ARM Cortex-A72 (Raspberry Pi 4), the acoustic environment poses three fundamental engineering hurdles:
1. **Severe Edge Resource Constraints:** The Raspberry Pi has limited CPU cycles and a constrained thermal envelope. Running continuous deep neural network inference on continuous raw audio streams would induce thermal throttling and high latency.
2. **Acoustic "Shortcut Learning" & Microphone Overfitting:** Neural networks tend to exploit spurious correlations. Different recording devices introduce distinct high-frequency noise floors, analog-to-digital converter (ADC) colorations, and sampling rate artifacts. Without principled frequency bounding and normalization, a convolutional neural network (CNN) easily learns the "fingerprint" of the recording hardware rather than the acoustic formant structure of infant vocal tracts.
3. **The False Positive Hazard in Edge Deployment:** In a real crib setting, transient impulse noises (such as hand claps, doors slamming, footsteps, coughing, or dropped objects) can trigger false alarms. Conversely, dead silence with floating-point numerical noise can be magnified into false activations if normalization is dynamic.

To overcome these challenges, the preprocessing system is bifurcated into two mutually consistent domains:
* **The Offline Pipeline (Dataset Standardization & Feature Generation):** Implemented in `prepare_raw_data.py`, `preprocessor.py`, and `data_loader.py`. It guarantees data uniformity, eliminates format discrepancies across heterogeneous datasets, prevents data leakage by strict stratified splitting prior to array generation, and extracts static-dimension 2D Log-Mel representations.
* **The Online Real-Time Pipeline (Low-Power Edge DSP & Gated Inference):** Implemented in `main_system.py` and `evaluate_local_pi.py`. It utilizes zero-copy ring buffering, anti-aliasing polyphase resampling, dual microsecond DSP pre-filtering gates (Root-Mean-Square Energy Gate & Spectrogram Peak-to-Average Power Ratio Gate), and fixed-range normalization before feeding quantized INT8 models.

```
+----------------------------------------------------------------------------------------------------+
|                                    AUDIO PREPROCESSING PIPELINE                                    |
+----------------------------------------------------------------------------------------------------+
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
      [ OFFLINE BATCH PIPELINE ]                                      [ REAL-TIME EDGE STREAM ]
 (prepare_raw_data.py + preprocessor.py)                                 (main_system.py)
                 │                                                                 │
     Heterogeneous Audio Files                                         Live USB Microphone Stream
  (.wav, .mp4, .webm, .ogg, .flac)                                    (Native Rate: e.g., 44.1 kHz)
                 │                                                                 │
                 ▼                                                                 ▼
      Monophonic Downmixing                                           Circular Ring Buffer (7.0 s)
                 │                                                                 │
                 ▼                                                                 ▼
      16 kHz Sinc Resampling                                          Polyphase Anti-Aliasing Resample
                 │                                                    (44.1 kHz -> 16.0 kHz)
                 ▼                                                                 │
   Fixed Duration Chunking (7.0 s)                                                 ▼
        (112,000 Samples)                                            DSP Pre-Filter Gate 1: RMS Energy
                 │                                                    [ Reject if RMS < 0.015 ]
                 │                                                                 │ (Passed)
                 │                                                                 ▼
                 │                                                   Short-Time Fourier Transform
                 │                                                   (N_fft=1024, Hop=256, Hann)
                 │                                                                 │
                 │                                                                 ▼
                 │                                                    Mel-Scale Filterbank (64 Bins)
                 │                                                    [ Strictly capped at fmax=4000 Hz ]
                 │                                                                 │
                 │                                                                 ▼
                 │                                                    Log-Power Decibel Scaling
                 │                                                    [ Pref = 1.0, top_db = None ]
                 │                                                                 │
                 │                                                                 ▼
                 │                                                   DSP Pre-Filter Gate 2: PAPR Gate
                 │                                                   [ Reject Transients if PAPR > 12.0 ]
                 │                                                                 │ (Passed)
                 ▼                                                                 ▼
   Fixed-Range Absolute Normalization:  [-100 dB, +55 dB]  ───>  [0.0, 1.0]
                 │                                                                 │
                 ▼                                                                 ▼
   Tensor Reshaping: (64, 438, 1) float32                           Tensor Reshaping: (1, 64, 438, 1) float32
                 │                                                                 │
                 ▼                                                                 ▼
      Train/Val/Test Datasets                                         Cascaded TFLite Edge Inference
       (Stage 1 & Stage 2 CNN)                                         (Stage 1 Gatekeeper -> Stage 2)
```

---

## 2. End-to-End Mathematical Formulations & Component Analysis

Each stage of the preprocessing pipeline is governed by rigorous mathematical principles from Digital Signal Processing (DSP) and psychoacoustics. Below is an exhaustive breakdown of every operational phase.

---

### Phase 1: Channel Downmixing (Monophonic Conversion)

#### 1. Mathematical Formulation
Given an arbitrary input audio recording with $C$ discrete audio channels $x_c[n]$ for $c \in \{1, 2, \dots, C\}$ and discrete time index $n$:
$$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^{C} x_c[n]$$
For standard stereophonic data ($C = 2$, Left $x_L[n]$ and Right $x_R[n]$):
$$x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$$

#### 2. Code Implementation Reference
* `prepare_raw_data.py`: `librosa.load(str(filepath), sr=sr, mono=True)` and ffmpeg fallback `-ac 1`.
* `main_system.py`: `AudioBuffer._audio_callback`: `new_audio = indata[:, 0].copy()`.

#### 3. Primary Objective
To eliminate inter-channel phase differences, spatial panning artifacts, and channel allocation discrepancies present across multi-source datasets (e.g., YouTube recordings, smartphone video containers, and studio microphones).

#### 4. Impact on the Overall Pipeline
* **Computational Complexity:** Decreases sample volume by $50\%$ compared to stereo data, halving memory bandwidth and processing requirements throughout all subsequent transformations.
* **Invariance:** Prevents the convolutional kernels from attempting to extract spatial orientation features, focusing feature extraction exclusively on spectral voice timbre and vocal tract resonance.

---

### Phase 2: Anti-Aliasing Resampling to $f_s = 16{,}000\text{ Hz}$

#### 1. Mathematical Formulation
A continuous-time audio signal $x(t)$ contains spectral content across frequency $f$. According to the **Nyquist-Shannon Sampling Theorem**, exact bandlimited reconstruction without aliasing requires the sampling frequency $f_s$ to satisfy:
$$f_s \ge 2 \cdot f_{\text{max\_content}}$$
where the **Nyquist Frequency** is defined as:
$$f_{\text{Nyquist}} = \frac{f_s}{2} = \frac{16{,}000\text{ Hz}}{2} = 8{,}000\text{ Hz}$$

If an audio stream sampled at native rate $f_{\text{native}}$ (e.g., $44{,}100\text{ Hz}$ or $48{,}000\text{ Hz}$) is decimated without filtering, any spectral component $f > f_{\text{Nyquist}}$ folds back into the lower frequency spectrum as an alias:
$$f_{\text{alias}} = |f - k \cdot f_s|$$
This injects artificial harmonic distortion into the lower vocal bands.

To prevent aliasing, resampling is formulated as a rational factor conversion $\frac{P}{Q}$:
$$\frac{P}{Q} = \frac{f_{\text{target}}}{f_{\text{native}}} = \frac{16{,}000}{44{,}100} = \frac{160}{441}$$

In `main_system.py`, this is executed via **Polyphase Anti-Aliasing Resampling** (`scipy.signal.resample_poly`):
1. **Zero-Stuffing Upsampling by $P$:**
   $$v[m] = \begin{cases} x[m/P], & m = 0, \pm P, \pm 2P, \dots \\ 0, & \text{otherwise} \end{cases}$$
2. **Bandlimiting via Finite Impulse Response (FIR) Sinc Filter:**
   $$h[m] = \frac{\sin(\pi m / M)}{\pi m / M} \cdot w_{\text{Kaiser}}[m], \quad M = \max(P, Q)$$
   with normalized cutoff frequency:
   $$\omega_c = \frac{\pi}{\max(P, Q)} = \frac{\pi}{441}$$
3. **Decimation by $Q$:**
   $$y[k] = (v * h)[k \cdot Q] = \sum_{m=-\infty}^{\infty} v[m] \cdot h[k \cdot Q - m]$$
4. **Polyphase Partitioning:** The filter $h[m]$ is partitioned into $P$ sub-filters $E_p[l] = h[l \cdot P + p]$:
   $$y[k] = \sum_{p=0}^{P-1} \sum_{l} x[l] \cdot E_p[k \cdot Q - l \cdot P]$$
   This guarantees that zero-valued samples are never computed, reducing floating-point operations by a factor of $P$.

```
Continuous Audio x(t)
         │
         ▼
[ Sampling: fs = 44.1 kHz ]  ──>  Bandwidth = 22.05 kHz
         │
         ▼
[ Anti-Aliasing Low-Pass Filter: Cutoff fc = 8.0 kHz ]
         │ (Removes frequencies above 8.0 kHz)
         ▼
[ Rational Polyphase Resampler: Up 160, Down 441 ]
         │
         ▼
Resampled Signal x[n] (fs = 16.0 kHz)  ──>  Nyquist Limit = 8.0 kHz (Zero Aliasing Distortion)
```

#### 2. Code Implementation Reference
* `prepare_raw_data.py`: `librosa.load(..., sr=16000)` & `ffmpeg -ar 16000`.
* `preprocessor.py`: `AudioPreprocessor.load_and_resample()` (`sr=16000`).
* `main_system.py`: `SpectrogramProcessor.process()` (`signal.resample_poly(native_audio, up=16000, down=native_rate)`).

#### 3. Primary Objective
* Standardize the temporal grid of all acoustic recordings to an identical temporal reference frame.
* Discard redundant ultra-high frequencies ($> 8\text{ kHz}$) that carry zero physiological information regarding infant crying.
* Provide an anti-aliased stream on edge devices without consuming excessive CPU power.

#### 4. Impact on the Overall Pipeline
* **Dimensional Reduction:** Compared to $44.1\text{ kHz}$ or $48\text{ kHz}$, downsampling to $16\text{ kHz}$ compresses the raw data buffer from $308{,}700$ samples down to exactly $112{,}000$ samples for a 7-second window ($63.7\%$ reduction in raw data volume).
* **Speed:** `signal.resample_poly` runs approximately **$100\times$ faster** than standard `librosa.resample` (which relies on continuous sinc interpolation via scipy/soxr), maintaining real-time execution on the Raspberry Pi 4 CPU without dropping microphone audio frames.

---

### Phase 3: Fixed Temporal Windowing (Padding & Slicing to $7.0\text{ s}$)

#### 1. Mathematical Formulation
Let $x[n]$ be an arbitrary discrete signal of length $L \in \mathbb{N}$ samples at $f_s = 16{,}000\text{ Hz}$.
The target sample count $N_{\text{target}}$ is defined by:
$$N_{\text{target}} = f_s \times T_{\text{duration}} = 16{,}000 \times 7.0 = 112{,}000 \text{ samples}$$

The framing operator $\mathcal{T}(x[n])$ behaves as follows:

$$\mathcal{T}(x[n]) = \begin{cases} 
x[0 : N_{\text{target}}], & \text{if } L = N_{\text{target}} \\
\left\{ x[k \cdot N_{\text{target}} : (k+1) \cdot N_{\text{target}}] \right\}_{k=0}^{\lfloor L / N_{\text{target}} \rfloor - 1}, & \text{if } L > N_{\text{target}} \\
x[n] \mathbin{\Vert} \mathbf{0}_{N_{\text{target}} - L}, & \text{if } L < N_{\text{target}}
\end{cases}$$
where $\mathbin{\Vert}$ denotes vector concatenation, and $\mathbf{0}_p$ denotes a zero vector of length $p$:
$$x_{\text{padded}}[n] = \begin{cases} x[n], & 0 \le n < L \\ 0, & L \le n < N_{\text{target}} \end{cases}$$

Trailing residual samples $R = L - \lfloor L / N_{\text{target}} \rfloor \cdot N_{\text{target}}$ for files where $L > N_{\text{target}}$ are safely discarded to avoid partial unrepresentative acoustic segments.

#### 2. Code Implementation Reference
* `prepare_raw_data.py`: `process_file(y, src_stem, out_dir, logger)`.
* `preprocessor.py`: `AudioPreprocessor.pad_or_truncate()`.
* `main_system.py`: `SpectrogramProcessor.process()` and `AudioBuffer` ring buffer (`maxlen=112000`).

#### 3. Primary Objective
* Enforce **strict dimensional stationarity**. Deep learning convolutional layers and hardware tensor accelerators (TFLite INT8 / XNNPACK) require rigid tensor shapes $(N, H, W, C)$.
* Capture the natural biological periodicity of infant distress. Infant cries consist of alternating expiratory cry phonations ($1.0 - 2.5\text{ s}$), inspiratory gasps ($0.2 - 0.5\text{ s}$), and rest intervals. A 7.0-second window captures at least $2$ to $3$ full respiratory cry cycles, providing the temporal depth required by Stage 2 to distinguish pain (abrupt, protracted high-energy phonations) from hunger (rhythmic, repetitive crescendos).

#### 4. Impact on the Overall Pipeline
* **Zero Tensor Re-allocation:** In edge inference, the input buffer is statically pre-allocated in RAM. Every forward pass executes on an identical memory block, eliminating dynamic memory allocations, garbage collection pauses, and memory fragmentation.
* **Dataset Expansion:** During offline preprocessing (`prepare_raw_data.py`), slicing long audio recordings into 7-second chunks acts as an effective non-destructive dataset multiplier, turning multi-minute recordings into dozens of high-quality independent training instances.

---

### Phase 4: Short-Time Fourier Transform (STFT) & Time-Frequency Spectral Analysis

#### 1. Mathematical Formulation
Audio signals are inherently non-stationary; their frequency content changes dynamically over time. The Short-Time Fourier Transform (STFT) divides the 1D signal into short, overlapping quasi-stationary segments and maps them into the complex time-frequency domain.

##### A. Hann Windowing
To eliminate spectral leakage (the appearance of spurious high-frequency components caused by sharp discontinuities at frame boundaries), a symmetric **Hann window** $w[m]$ of length $N_{\text{fft}} = 1024$ is applied:
$$w[m] = 0.5 \left( 1 - \cos\left( \frac{2\pi m}{N_{\text{fft}} - 1} \right) \right) = \sin^2\left( \frac{\pi m}{N_{\text{fft}} - 1} \right), \quad 0 \le m \le N_{\text{fft}}-1$$

##### B. Discrete Short-Time Fourier Transform Equation
For time frame index $t \in \{0, 1, \dots, T_{\text{frames}}-1\}$ and discrete frequency bin index $k \in \{0, 1, \dots, \frac{N_{\text{fft}}}{2}\}$:
$$X(t, k) = \sum_{m=0}^{N_{\text{fft}}-1} x[t \cdot H + m] \cdot w[m] \cdot \exp\left( -j \frac{2\pi k m}{N_{\text{fft}}} \right)$$
where:
* $N_{\text{fft}} = 1024$ samples (Window length $\approx \frac{1024}{16{,}000} = 64\text{ ms}$).
* $H = \text{hop\_length} = 256$ samples (Frame step $\Delta t = \frac{256}{16{,}000} = 16\text{ ms}$).
* Window overlap percentage:
  $$\text{Overlap} = \left( 1 - \frac{H}{N_{\text{fft}}} \right) \times 100\% = \left( 1 - \frac{256}{1024} \right) \times 100\% = 75\%$$

##### C. Frequency and Time Resolution Trade-off
By the Gabor-Heisenberg Uncertainty Principle:
$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$
* **Frequency Resolution per Bin:**
  $$\Delta f = \frac{f_s}{N_{\text{fft}}} = \frac{16{,}000\text{ Hz}}{1024} = 15.625\text{ Hz}$$
* **Temporal Resolution per Frame:**
  $$\Delta t = \frac{H}{f_s} = \frac{256}{16{,}000} = 0.016\text{ s} = 16.0\text{ ms}$$

##### D. Linear Power Spectrogram
The phase information is discarded because human auditory perception is largely phase-deaf in environmental audio classification. The linear power spectrum $P(t, k)$ is computed by the squared magnitude:
$$P(t, k) = |X(t, k)|^2 = \text{Re}(X(t, k))^2 + \text{Im}(X(t, k))^2$$

##### E. Temporal Output Dimension Derivation
With centered windowing (`center=True`), the signal is padded by $N_{\text{fft}} / 2 = 512$ samples on both sides. The exact number of time frames $T_{\text{frames}}$ is:
$$T_{\text{frames}} = 1 + \left\lfloor \frac{N_{\text{target}}}{H} \right\rfloor = 1 + \left\lfloor \frac{112{,}000}{256} \right\rfloor = 1 + 437 = 438 \text{ frames}$$
The linear spectrogram matrix dimensions are:
$$\text{Shape}(P) = \left( \frac{N_{\text{fft}}}{2} + 1, \, T_{\text{frames}} \right) = (513, \, 438)$$

#### 2. Code Implementation Reference
* `preprocessor.py`: `librosa.feature.melspectrogram(..., n_fft=1024, hop_length=256)`.
* `main_system.py`: `SpectrogramProcessor.process()`.

#### 3. Primary Objective
Transform raw 1D time-domain waveforms into a 2D structured representation where acoustic formants, harmonic stacks, and pitch contours can be recognized as spatial visual textures by 2D Convolutional Neural Networks.

#### 4. Impact on the Overall Pipeline
* $\Delta f = 15.625\text{ Hz}$ provides enough resolution to cleanly separate the fundamental frequency $F_0$ of an infant cry ($350 - 700\text{ Hz}$) from its 2nd, 3rd, and 4th harmonics.
* $75\%$ frame overlap ensures smooth temporal transitions without spectral aliasing across frame boundaries.

---

### Phase 5: Mel-Scale Filterbank Integration & High-Frequency Capping ($f_{\text{max}} = 4{,}000\text{ Hz}$)

#### 1. Mathematical Formulation
Human frequency perception is non-linear; the human auditory system discriminates pitch differences much more accurately at low frequencies than at high frequencies. The **Mel Scale** maps physical frequencies in Hertz to perceived pitch.

##### A. Hertz-to-Mel and Mel-to-Hertz Mapping
Using the O'Shaughnessy formulation:
$$m = 2595 \cdot \log_{10}\left( 1 + \frac{f}{700} \right) = 1127 \cdot \ln\left( 1 + \frac{f}{700} \right)$$
$$f = 700 \cdot \left( 10^{\frac{m}{2595}} - 1 \right) = 700 \cdot \left( e^{\frac{m}{1127}} - 1 \right)$$

##### B. High-Frequency Bounding ($f_{\text{min}} = 0\text{ Hz}, f_{\text{max}} = 4{,}000\text{ Hz}$)
The filterbank is strictly bounded within $[0, 4000]\text{ Hz}$:
$$m_{\text{min}} = m(0) = 0\text{ Mels}$$
$$m_{\text{max}} = m(4000) = 2595 \cdot \log_{10}\left( 1 + \frac{4000}{700} \right) \approx 2146.06\text{ Mels}$$

##### C. Triangular Filterbank Construction
For $n_{\text{mels}} = 64$, an array of $M + 2 = 66$ linearly spaced points in Mel space is constructed:
$$m_i = m_{\text{min}} + i \cdot \frac{m_{\text{max}} - m_{\text{min}}}{M + 1}, \quad i \in \{0, 1, \dots, M+1\}$$
These $m_i$ values are converted back to linear Hertz $f_i = f(m_i)$, and mapped to the closest discrete STFT bin indices $k_i$:
$$k_i = \left\lfloor \frac{N_{\text{fft}} + 1}{f_s} \cdot f_i \right\rfloor = \left\lfloor \frac{1025}{16{,}000} \cdot f_i \right\rfloor$$

Each triangular filter $H_m[k]$ for $m \in \{1, 2, \dots, 64\}$ is defined over linear bins $k \in \{0, \dots, 512\}$ as:
$$H_m[k] = \begin{cases} 
0, & k < k_{m-1} \\
\dfrac{k - k_{m-1}}{k_m - k_{m-1}}, & k_{m-1} \le k \le k_m \\
\dfrac{k_{m+1} - k}{k_{m+1} - k_m}, & k_m \le k \le k_{m+1} \\
0, & k > k_{m+1}
\end{cases}$$

##### D. Filterbank Multiplication
The Mel Spectrogram $S_{\text{mel}}(m, t)$ is computed by multiplying the linear power spectrogram $P(k, t)$ with the Mel filterbank matrix:
$$S_{\text{mel}}(m, t) = \sum_{k=0}^{N_{\text{fft}}/2} H_m[k] \cdot P(t, k)$$
This compresses the frequency axis from $513$ linear bins down to $64$ Mel bins:
$$\text{Shape}(S_{\text{mel}}) = (64, \, 438)$$

```
Frequency Spectrum (Linear: 0 Hz to 8000 Hz)
[--- 0 Hz to 4000 Hz: Infant Formants & Cry Harmonics ---] [--- 4001 Hz to 8000 Hz: Noise & Mic Artifacts ---]
                            │                                                              │
                            ▼                                                              ▼
               64 Dense Mel-Scale Filters                                      TRUNCATED / EXCLUDED (fmax=4000)
                            │
                            ▼
         Output Mel Matrix: (64 Mel Bins x 438 Time Frames)
```

#### 2. Code Implementation Reference
* `stage_one_config.py` & `stage_two_config.py`: `n_mels = 64`, `fmax = 4000`.
* `preprocessor.py`: `AudioPreprocessor.to_log_mel_spectrogram()`.
* `main_system.py`: `SpectrogramProcessor.process()`.

#### 3. Primary Objective & Acoustic Rationale
* **Formant Matching:** In human infants, the vocal tract length is approximately $7 - 8\text{ cm}$. This produces fundamental cry frequencies ($F_0$) between $350\text{ Hz}$ and $700\text{ Hz}$, with the first three critical formants ($F_1, F_2, F_3$) residing between $1{,}100\text{ Hz}$ and $3{,}800\text{ Hz}$. Energy above $4{,}000\text{ Hz}$ consists almost entirely of unvoiced breath friction, ambient hiss, or room reverberation.
* **Overcoming "Shortcut Learning":** Different recording devices (cheap electret microphones vs studio equipment) have drastically diverging frequency roll-offs and ADC noise profiles above $4\text{ kHz}$. If the network observed the $[4000, 8000]\text{ Hz}$ band, it would overfit to the hardware noise floor of specific audio clips rather than infant pathology. Capping at $f_{\text{max}} = 4000\text{ Hz}$ forces the CNN to evaluate only genuine vocal tract resonances.

#### 4. Impact on the Overall Pipeline
* **Dimensional Compression:** Compressing $513$ linear bins to $64$ Mel bands yields an **$87.5\%$ reduction** in spectral feature dimensions.
* **Model Size Reduction:** The flattened feature vector feeding the dense classification head is drastically smaller, enabling the entire CNN to remain under **$540{,}000$ parameters** ($< 2.1\text{ MB}$ unquantized, $< 550\text{ KB}$ INT8 quantized).

---

### Phase 6: Logarithmic Power Transformation (Decibel Scaling)

#### 1. Mathematical Formulation
Acoustic power spans multiple orders of magnitude. The human ear perceives loudness logarithmically according to the **Weber-Fechner Law**:
$$\text{Perceived Loudness} \propto \log(\text{Physical Power})$$

The transformation converts the linear Mel power $S_{\text{mel}}(m, t)$ to the decibel (dB) scale:
$$S_{\text{dB}}(m, t) = 10 \cdot \log_{10}\left( \frac{\max(\epsilon, \, S_{\text{mel}}(m, t))}{P_{\text{ref}}} \right)$$
where:
* $P_{\text{ref}} = 1.0$ (Strict absolute power reference).
* $\epsilon = 10^{-10}$ (Floating-point stabilization floor to prevent $\log_{10}(0) = -\infty$).
* Dynamic range limiting: `top_db = None` (No adaptive per-sample dynamic range thresholding).

#### 2. Code Implementation Reference
* `preprocessor.py`: `librosa.power_to_db(mel, ref=1.0, top_db=None)`.
* `main_system.py`: `SpectrogramProcessor.process()`.

#### 3. Primary Objective & Architectural Defense
* In typical audio processing libraries, the default setting is `ref=np.max` and `top_db=80.0`. In that mode, every single file is normalized relative to its own local peak.
* **The Dangerous Pitfall of Relative Scaling:** In a nursery environment, during periods of absolute silence, the peak value might be an imperceptible background hiss of $10^{-7}$. If relative normalization (`ref=np.max`) were used, that microscopic hiss would be boosted to $0\text{ dB}$, inflating ambient noise into loud-looking visual textures on the spectrogram. This was the direct root cause of **false positive crying detections on quiet rooms**.
* By locking $P_{\text{ref}} = 1.0$ and `top_db = None`, absolute acoustic energy is preserved:
  * Digital silence: $S_{\text{dB}} \approx 10 \cdot \log_{10}(10^{-10} / 1.0) = -100\text{ dB}$.
  * Quiet ambient room noise: $S_{\text{dB}} \approx -60\text{ dB}$ to $-40\text{ dB}$.
  * Adult conversation: $S_{\text{dB}} \approx -20\text{ dB}$ to $0\text{ dB}$.
  * Loud infant distress cry: $S_{\text{dB}} \approx +15\text{ dB}$ to $+55\text{ dB}$.

#### 4. Impact on the Overall Pipeline
This step stabilizes the numeric distribution feeding the normalization phase, guaranteeing that low-energy acoustic states remain numerically separated from high-energy vocalizations.

---

### Phase 7: Fixed-Range Absolute Normalization ($[-100, +55]\text{ dB} \to [0, 1]$)

#### 1. Mathematical Formulation
Neural network parameters optimize best when input features are bounded within a stable numerical interval $[0, 1]$ or $[-1, 1]$. 

Instead of statistical Z-score standardization ($z = \frac{x - \mu}{\sigma}$) or dynamic per-sample min-max scaling, the pipeline employs **Static Range Affine Mapping with Saturation (Clipping)**:

$$S_{\text{norm}}(m, t) = \text{clip}\left( \frac{S_{\text{dB}}(m, t) - \text{Min}_{\text{dB}}}{\text{Max}_{\text{dB}} - \text{Min}_{\text{dB}}}, \, 0.0, \, 1.0 \right)$$
Substituting the empirically established acoustic bounds:
$$\text{Min}_{\text{dB}} = -100.0\text{ dB}, \quad \text{Max}_{\text{dB}} = +55.0\text{ dB}$$
$$\text{Dynamic Span} = 55.0 - (-100.0) = 155.0\text{ dB}$$

$$\boxed{S_{\text{norm}}(m, t) = \min\left( \max\left( \frac{S_{\text{dB}}(m, t) + 100.0}{155.0}, \, 0.0 \right), \, 1.0 \right)}$$

```
Decibel Scale: -100 dB (Silence Floor) ──────── -45 dB (Ambient) ──────── 0 dB (Speech) ──────── +55 dB (Peak Cry)
                          │                          │                     │                     │
                          ▼                          ▼                     ▼                     ▼
Normalized Scale:        0.00                       0.35                  0.64                  1.00
```

#### 2. Numerical Regime Mapping Table
| Acoustic Regime | Approximate Sound Level ($S_{\text{dB}}$) | Normalized Output ($S_{\text{norm}}$) | Neural Interpretation |
| :--- | :--- | :--- | :--- |
| **Digital Silence / Disconnected Mic** | $\le -100\text{ dB}$ | $\mathbf{0.0000}$ | Zero activation in initial convolutional filters |
| **Quiet Nursery Background** | $-50\text{ dB} \text{ to } -40\text{ dB}$ | $\mathbf{0.3225} \text{ to } \mathbf{0.3871}$ | Baseline ambient offset; insufficient to trigger cry kernels |
| **Moderate Environmental Noise / TV** | $-20\text{ dB} \text{ to } -5\text{ dB}$ | $\mathbf{0.5161} \text{ to } \mathbf{0.6129}$ | Distinguishable non-cry acoustic activity |
| **Vigorous Infant Cry (Expiratory)** | $+20\text{ dB} \text{ to } +55\text{ dB}$ | $\mathbf{0.7742} \text{ to } \mathbf{1.0000}$ | Saturated high-energy formant patterns |

#### 3. Code Implementation Reference
* `preprocessor.py`: `AudioPreprocessor.normalize()`:
  `return np.clip((spectrogram + 100.0) / 155.0, 0.0, 1.0)`
* `main_system.py`: `SpectrogramProcessor.process()`.

#### 4. Primary Objective & Comparative Advantages
* **Elimination of Data Leakage:** Standard normalization methods (like computing global mean $\mu$ and standard deviation $\sigma$ across the training set) introduce statistical dependencies. If the testing environment has a different acoustic bias, the precomputed statistics become misaligned. Fixed-range normalization is **strictly stateless and deterministic**.
* **Preservation of Loudness Dynamics:** In infant cry analysis, sound volume is a primary clinical indicator: pain cries exhibit explosive peak acoustic energy compared to tired whimpers. Dynamic min-max scaling flattens loud and soft cries to the exact same range, destroying loudness information. Fixed-range normalization preserves these volume dynamics.
* **INT8 Quantization Friendliness:** The output tensor is strictly bounded in $[0.0, 1.0]$ with no extreme outliers. This aligns with symmetric/asymmetric 8-bit integer quantization scales in TensorFlow Lite:
  $$q = \text{round}\left( \frac{S_{\text{norm}}}{S_{\text{scale}}} \right) + Z_{\text{zero\_point}}$$
  yielding minimal quantization noise and zero clipping distortion during edge execution.

---

### Phase 8: Online Edge DSP Gating (Dual-Stage Low-Power Pre-Filtering)

To guarantee that the Raspberry Pi 4 does not waste processor cycles running neural network inference on silence or non-cry transient noises, `main_system.py` introduces a **Two-Stage Hardware-Protection DSP Gate**.

```
[ Incoming 7-second Native Audio Buffer ]
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  GATE 1: Root Mean Square (RMS)     │
  │  RMS = sqrt( mean( x[n]^2 ) )       │
  └─────────────────────────────────────┘
                     │
         RMS < 0.015 ?
        ┌────────────┴────────────┐
     YES│                         │NO (Energy Present)
        ▼                         ▼
 [ REJECT: SILENCE ]    [ Compute Log-Mel Spectrogram ]
 (Skip AI Inference)              │
                                  ▼
                        ┌─────────────────────────────────────┐
                        │  GATE 2: Spectrogram PAPR           │
                        │  PAPR = max(E(t)) / mean(E(t))      │
                        └─────────────────────────────────────┘
                                  │
                             PAPR > 12.0 ?
                            ┌─────┴─────┐
                         YES│           │NO (Sustained Cry Formants)
                            ▼           ▼
                   [ REJECT: IMPULSE ]  [ PASS TO STAGE 1 CNN ]
                   (Clap, Slam, Knock)  (Execute TFLite Inference)
```

---

#### Gate 1: Root-Mean-Square (RMS) Energy Gate

##### 1. Mathematical Formulation
Given the 7.0-second discrete audio buffer $x[n]$ of length $N = 112{,}000$ samples:
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}$$
The gating decision rule is:
$$\mathcal{G}_{\text{RMS}}(x) = \begin{cases} 
\text{Reject (Skip Inference)}, & \text{if } \text{RMS} < \tau_{\text{RMS}} \\
\text{Proceed to STFT}, & \text{if } \text{RMS} \ge \tau_{\text{RMS}}
\end{cases}$$
where the empirical threshold is configured in `main_system.py` as:
$$\tau_{\text{RMS}} = 0.015$$

##### 2. Physical and Acoustic Motivation
* USB microphone background noise floor in a silent nursery produces an $\text{RMS} \approx 0.005 - 0.008$.
* A faint baby cry or distress vocalization yields an $\text{RMS} \ge 0.025 - 0.060$.
* Peak amplitude $\max(|x[n]|)$ is vulnerable to single-sample digital spikes, electrostatic clicks, or ADC pops. RMS measures the true **sustained physical energy** integrated across the entire 7-second window.

##### 3. Computational Impact
Computing $\text{RMS}$ requires only $N$ multiply-accumulate operations in vector registers (SIMD / NEON on ARM Cortex-A72), completing in **$< 0.15\text{ milliseconds}$**. If the crib is quiet, the system goes back to sleep immediately without computing STFTs or waking the neural network.

---

#### Gate 2: Spectrogram Peak-to-Average Power Ratio (PAPR) Gate

##### 1. Mathematical Formulation
Transient acoustic impulses (such as door slams, book drops, hand clapping, or coughing) can exhibit high instantaneous energy that passes Gate 1. However, their energy is concentrated in a tiny fraction of time ($10 - 50\text{ ms}$), unlike infant cries which are sustained vocalizations ($1.5 - 3.5\text{ s}$).

Gate 2 computes the **Peak-to-Average Power Ratio (PAPR)** across the temporal energy envelope of the 2D spectrogram $S_{\text{norm}}(m, t) \in \mathbb{R}^{M \times T}$ ($M=64, T=438$):
1. **Compute Mean Temporal Energy Envelope:**
   $$E[t] = \frac{1}{M} \sum_{m=0}^{M-1} S_{\text{norm}}(m, t), \quad \forall t \in \{0, 1, \dots, T-1\}$$
2. **Compute Global Temporal Mean Energy:**
   $$\bar{E} = \frac{1}{T} \sum_{t=0}^{T-1} E[t]$$
3. **Calculate Spectrogram PAPR:**
   $$\text{PAPR} = \begin{cases} \dfrac{\max_{t \in [0, T-1]} E[t]}{\bar{E}}, & \text{if } \bar{E} > 10^{-6} \\ 0.0, & \text{otherwise} \end{cases}$$
4. **The Gating Decision Rule:**
   $$\mathcal{G}_{\text{PAPR}}(S) = \begin{cases} 
   \text{Reject (Impulsive Noise)}, & \text{if } \text{PAPR} > \tau_{\text{PAPR}} \\
   \text{Proceed to Stage 1 CNN}, & \text{if } \text{PAPR} \le \tau_{\text{PAPR}}
   \end{cases}$$
   where the threshold configured in `main_system.py` is:
   $$\tau_{\text{PAPR}} = 12.0$$

##### 2. Acoustic Characteristics Comparison
* **Impulsive Noise (Clap, Slam, Dropped Object):** Energy spikes in $1 - 2$ time frames ($t \in [t_0, t_0+2]$) while remaining near zero everywhere else.
  $$\max(E[t]) \approx 0.85, \quad \bar{E} \approx 0.03 \implies \text{PAPR} \approx \frac{0.85}{0.03} \approx 28.3 \quad (\gg 12.0) \implies \mathbf{REJECTED}$$
* **Infant Cry Vocalization:** Energy is distributed across sustained expiratory cries and harmonic formants.
  $$\max(E[t]) \approx 0.75, \quad \bar{E} \approx 0.18 \implies \text{PAPR} \approx \frac{0.75}{0.18} \approx 4.16 \quad (\ll 12.0) \implies \mathbf{PASSED}$$

##### 3. System Impact
Completely eliminates transient false alarms at the edge before the Stage 1 binary classifier is invoked, protecting the mechanical rocking motor from triggering on accidental room bumps or loud footsteps.

---

## 3. Comprehensive Parameter & Mathematical Specification Table

The following table summarizes every single constant, hyperparameter, transformation step, and dimensional transition across the entire preprocessing lifecycle:

| Pipeline Stage | Parameter / Symbol | Exact Value | Mathematical Function / Domain | Code Location | System Rationale & Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio Acquisition** | $f_{\text{native}}$ | $44{,}100\text{ Hz}$ or $48{,}000\text{ Hz}$ | Raw discrete time domain $x_{\text{raw}}[n]$ | `main_system.py:109` | Hardware standard for plug-and-play USB microphones. |
| **Monophonic Downmixing** | $C_{\text{out}}$ | $1\text{ channel}$ | $x[n] = \frac{1}{C}\sum x_c[n]$ | `prepare_raw_data.py:85` | Halves processing volume; discards spatial orientation noise. |
| **Resampling** | $f_s$ | $16{,}000\text{ Hz}$ | Polyphase FIR filter $\frac{P}{Q} = \frac{160}{441}$ | `stage_one_config.py:26` | Matches infant voice bandwidth; reduces data volume by $63.7\%$. |
| **Temporal Duration** | $T_{\text{sec}}$ | $7.0\text{ seconds}$ | Continuous window length | `stage_one_config.py:27` | Spans $2 - 3$ full respiration and cry cycles. |
| **Sample Count** | $N_{\text{target}}$ | $112{,}000\text{ samples}$ | $N = f_s \times T_{\text{sec}} = 16000 \times 7.0$ | `stage_one_config.py:28` | Static tensor allocation; required for INT8 quantization. |
| **STFT Window Length** | $N_{\text{fft}}$ | $1024\text{ samples}$ ($64\text{ ms}$) | $\Delta f = f_s / N_{\text{fft}} = 15.625\text{ Hz}$ | `stage_one_config.py:32` | Resolves individual harmonic overtones of vocal cords. |
| **STFT Hop Length** | $H$ | $256\text{ samples}$ ($16\text{ ms}$) | Frame stride; $75\%$ overlap | `stage_one_config.py:33` | Captures rapid formant transitions and cry onsets. |
| **Spectral Windowing** | $w[m]$ | Hann Window | $\sin^2(\pi m / (N_{\text{fft}}-1))$ | `librosa` default | Suppresses boundary discontinuities and spectral leakage. |
| **Mel Filter Count** | $n_{\text{mels}}$ | $64\text{ bins}$ | Non-linear psychoacoustic Mel scale | `stage_one_config.py:31` | Compresses frequency dimension by $87.5\%$ for edge CNN. |
| **Frequency Ceiling** | $f_{\text{max}}$ | $4{,}000\text{ Hz}$ | Hard cutoff of Mel filterbank | `stage_one_config.py:34` | **Critical:** Blocks microphone noise and prevents shortcut learning. |
| **Decibel Power Scale** | $P_{\text{ref}}$ | $1.0$ | $10 \cdot \log_{10}(P / 1.0)$ | `preprocessor.py:62` | Preserves true physical volume; prevents silence amplification. |
| **Decibel Floor Limit** | `top_db` | `None` | Unconstrained dynamic floor | `preprocessor.py:62` | Prevents per-sample dynamic thresholding from boosting hiss. |
| **Normalization Floor** | $\text{Min}_{\text{dB}}$ | $-100.0\text{ dB}$ | Digital silence lower bound | `preprocessor.py:74` | Maps near-zero acoustic power directly to $0.0$. |
| **Normalization Ceiling**| $\text{Max}_{\text{dB}}$ | $+55.0\text{ dB}$ | Intense cry upper bound | `preprocessor.py:74` | Maps peak crying energy to $1.0$ without clipping. |
| **Normalization Scale**  | $\Delta_{\text{range}}$| $155.0\text{ dB}$ | $(S_{\text{dB}} + 100.0) / 155.0$ | `preprocessor.py:74` | Bounded $[0.0, 1.0]$ affine transformation. |
| **DSP Gate 1 (RMS)**    | $\tau_{\text{RMS}}$ | $0.015$ | $\sqrt{\frac{1}{N}\sum x[n]^2} < 0.015$ | `main_system.py:93` | Rejects ambient silence in $< 0.15\text{ ms}$ on Raspberry Pi CPU. |
| **DSP Gate 2 (PAPR)**   | $\tau_{\text{PAPR}}$| $12.0$ | $\max(E[t]) / \bar{E} > 12.0$ | `main_system.py:100` | Rejects impulsive transients (hand claps, door slams, bumps). |
| **Final Tensor Output** | $\text{Shape}$ | $(64, 438, 1)$ | $(n_{\text{mels}}, T_{\text{frames}}, C_{\text{in}})$ | `preprocessor.py:94` | Exact shape required by Stage 1 & Stage 2 CNN architectures. |

---

## 4. Architectural Interactions with Downstream Stages

The preprocessing module does not operate in isolation; its design directly determines the behavior, convergence, and robustness of the downstream components:

```
[ Preprocessing Pipeline: Shape (64, 438, 1) ]
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ STAGE 1: GATEKEEPER CNN ]   [ STAGE 2: EXPERT CLASSIFIER ]
Binary Task: CRY vs NO_RESPONSE  Multi-Class: Pain vs Hungry vs Tired
Lightweight 4-Block Topology  Transfer Learning from Stage 1 Base
                       │
                       ▼
       [ INT8 Post-Training Quantization ]
       [ Edge Deployment on Raspberry Pi 4 ]
```

### 1. Interaction with Stage 1 (The Gatekeeper CNN)
* **Separation of Silence and Noise:** By preserving absolute energy dynamics via fixed-range normalization ($[-100, +55]\text{ dB}$), Stage 1 receives spectrograms where background noise is faint ($0.30 - 0.45$) and crying is intense ($0.80 - 1.00$). The binary classifier easily draws a decision boundary, achieving **$98\%+$ classification accuracy**.
* **Microphone Invariance:** The hard cutoff at $f_{\text{max}} = 4000\text{ Hz}$ ensures that Stage 1 does not overfit to specific high-frequency room characteristics or microphone recording flaws.

### 2. Interaction with Stage 2 (The Expert Classifier via Transfer Learning)
* **Preservation of Harmonic Micro-Structures:** Slicing at $7.0\text{ seconds}$ combined with a $16\text{ ms}$ STFT hop preserves the temporal pitch transitions ($F_0$ modulation) and phonation breaks characteristic of hunger vs pain.
* **Shared Acoustic Representations:** Because Stage 1 and Stage 2 use the exact same preprocessing configuration, the feature extraction layers of Stage 1 can be frozen and transferred directly to Stage 2 (`stage_two_pipeline.py`). Stage 2 learns to distinguish subtle infant cry nuances on a modest dataset without re-learning low-level acoustic primitives.

### 3. Interaction with Edge Quantization (TFLite INT8 Engine)
* **Bounded Input Domain:** Because every element of the input spectrogram is strictly bounded in $[0.0, 1.0]$, the TFLite quantization engine calculates optimal scale $S$ and zero-point $Z$ parameters with zero clipping saturation:
  $$S = \frac{1.0 - 0.0}{255} \approx 0.00392, \quad Z = -128 \text{ (for int8)}$$
* **Zero Run-Time Feature Drift:** In dynamic normalization approaches, slight variations in background noise cause the input distribution to drift, degrading quantized model accuracy. The stateless, fixed-range normalization eliminates distribution drift entirely.

---

## 5. Preventative Data Engineering: Data Leakage Elimination

A critical architectural achievement in this project is the strict prevention of **Data Leakage**:

1. **Split-First Architecture (`data_loader.py` & `stage_two_data_loader.py`):**
   * Many audio pipelines mistakenly compute dataset-wide statistics (mean, variance, min, max) across the entire corpus *before* splitting into Train, Validation, and Test sets. This leaks information from the test set into the model's training pipeline.
   * In this system, dataset discovery and stratified splitting occur **first** (`DataLoader.split_data()` using stratified 70% / 15% / 15% partitions).
2. **Stateless Preprocessing:**
   * The `AudioPreprocessor` class is **stateless**. It does not maintain or update any running dataset statistics. Each audio file is processed independently through deterministic mathematical functions ($f_{\text{max}}=4000$, $\text{ref}=1.0$, fixed range $[-100, 55]\text{ dB}$).
   * This guarantees that test set evaluations (`stage_one_evaluate.py`, `stage_two_evaluate.py`, `evaluate_local_pi.py`) represent genuine generalization performance.

---

## 6. Execution Lifecycle: Code Traceability Matrix

The table below maps each theoretical preprocessing operation directly to its implementation in the codebase:

| Conceptual Operation | Script | Class / Function | Exact Lines of Code |
| :--- | :--- | :--- | :--- |
| **Offline Data Standardization** | `prepare_raw_data.py` | `load_audio()`, `process_file()` | Lines 73–156 |
| **Stratified Dataset Splitting** | `data_loader.py` | `DataLoader.split_data()` | Lines 69–104 |
| **Audio Loading & Resampling** | `preprocessor.py` | `AudioPreprocessor.load_and_resample()` | Lines 33–36 |
| **Padding / Truncation** | `preprocessor.py` | `AudioPreprocessor.pad_or_truncate()` | Lines 38–44 |
| **STFT & Mel Filterbank** | `preprocessor.py` | `AudioPreprocessor.to_log_mel_spectrogram()` | Lines 46–60 |
| **Decibel Power Scaling** | `preprocessor.py` | `librosa.power_to_db(mel, ref=1.0)` | Line 62 |
| **Fixed-Range Normalization** | `preprocessor.py` | `AudioPreprocessor.normalize()` | Lines 65–74 |
| **Live Audio Capture** | `main_system.py` | `AudioBuffer._audio_callback()` | Lines 330–350 |
| **Polyphase Resampling** | `main_system.py` | `SpectrogramProcessor.process()` | Lines 374–385 |
| **Real-time RMS Energy Gate** | `main_system.py` | `SmartCribBrain._handle_listening()` | Lines 857–875 |
| **Real-time Spectrogram PAPR Gate**| `main_system.py` | `SmartCribBrain._handle_listening()` | Lines 881–906 |
| **TFLite Engine Ingestion** | `main_system.py` | `TFLiteEngine.predict()` | Lines 448–456 |

---

## 7. Summary & Key Conclusions

The preprocessing pipeline in the **Two-Stage Edge AI Infant Cry Classifier** exemplifies domain-specific signal processing engineered for embedded deep learning:
1. **Biological Realism:** Aligning window lengths ($7.0\text{ s}$) with infant respiration cycles and bounding frequency ($f_{\text{max}} = 4{,}000\text{ Hz}$) to vocal tract formants eliminates shortcut learning while preserving vital diagnostic features.
2. **Deterministic Stability:** Replacing adaptive dynamic normalization with absolute fixed-range scaling ($[-100, +55]\text{ dB} \to [0, 1]$) preserves physical loudness relationships and protects the system from amplifying room silence into false positives.
3. **Edge Optimization:** Incorporating microsecond DSP pre-filtering gates (RMS energy and Spectrogram PAPR) allows the Raspberry Pi 4 to suppress ambient silence and transient physical shocks without running neural network inference, maintaining low power consumption and reliable smart crib automation.

