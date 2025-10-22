Perfect — having a **supervised dataset** with clean source signals and mixtures gives you a great foundation to build and evaluate source separation models, even using **classic methods** like NMF or Wiener filtering.

Let me walk you through how you can use classic approaches like **supervised NMF** or **Wiener filtering** for single-channel voice separation.

---

## ✅ Step-by-Step: Supervised NMF for Single-Channel Voice Separation

### 🔧 Assumptions:

* You have access to:

  * Mixture audio (e.g., `mixture.wav`)
  * Clean sources (e.g., `speaker1.wav`, `speaker2.wav`)
* All signals are time-aligned and sampled equally.

---

## 🧠 Overview of Supervised NMF

You’ll use the clean sources to **learn spectral templates** (`W_speaker1`, `W_speaker2`) and then use those to decompose the **mixture spectrogram**.

### Steps:

1. Compute **magnitude spectrograms** of sources and mixture.
2. Train **NMF basis matrices (W)** from each speaker’s spectrogram.
3. Concatenate `W1` and `W2` to get a combined basis `W_combined`.
4. Use NMF to **only optimize H** (activations) for the mixture, keeping `W_combined` fixed.
5. Reconstruct each source using their respective parts of `W_combined` and `H`.

---

## 🧪 Python Example: Supervised NMF Using Librosa + Scikit-learn

Here's a basic example using `librosa` and `sklearn.decomposition.NMF`.

### ✅ Requirements:

```bash
pip install librosa scikit-learn numpy soundfile
```

### 🐍 Code:

```python
import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF

# --- Load clean sources and mixture ---
sr = 16000
y1, _ = librosa.load("speaker1.wav", sr=sr)
y2, _ = librosa.load("speaker2.wav", sr=sr)
mix, _ = librosa.load("mixture.wav", sr=sr)

# --- Compute magnitude spectrograms ---
def mag_spectrogram(y, n_fft=1024, hop_length=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return np.abs(S), np.angle(S)

S1_mag, _ = mag_spectrogram(y1)
S2_mag, _ = mag_spectrogram(y2)
S_mix_mag, S_mix_phase = mag_spectrogram(mix)

# --- Train NMF on clean sources to get W matrices ---
n_components = 30  # number of basis vectors per speaker

nmf1 = NMF(n_components=n_components, init='random', random_state=0, max_iter=300)
W1 = nmf1.fit_transform(S1_mag.T)
H1 = nmf1.components_

nmf2 = NMF(n_components=n_components, init='random', random_state=0, max_iter=300)
W2 = nmf2.fit_transform(S2_mag.T)
H2 = nmf2.components_

# --- Combine W matrices and run NMF on mixture with fixed W ---
W_combined = np.concatenate([H1, H2], axis=0).T  # Shape: (freq_bins, 2*n_components)

nmf_mix = NMF(n_components=2*n_components, init='custom', max_iter=200)
H_mix = nmf_mix.fit_transform(S_mix_mag.T, W=W_combined)
W_fixed = nmf_mix.components_

# --- Separate sources ---
# Source 1
S1_recon = np.dot(H_mix[:, :n_components], W_fixed[:n_components, :])
# Source 2
S2_recon = np.dot(H_mix[:, n_components:], W_fixed[n_components:, :])

# --- Convert back to time domain ---
def reconstruct_audio(S_mag, phase, hop_length=512):
    S_complex = S_mag.T * np.exp(1j * phase)
    return librosa.istft(S_complex, hop_length=hop_length)

y1_est = reconstruct_audio(S1_recon, S_mix_phase)
y2_est = reconstruct_audio(S2_recon, S_mix_phase)

# --- Save results ---
sf.write("est_speaker1.wav", y1_est, sr)
sf.write("est_speaker2.wav", y2_est, sr)
```

---

### 🔍 Notes:

* This assumes **magnitude spectrograms**, and phase is reused from the mixture. It’s not perfect, but works reasonably well.
* NMF components are shared between speakers (concatenated).
* Works best when speakers have **distinct spectral profiles**.

---

## 🔬 Evaluation

Since you have clean sources, you can evaluate using:

* **SDR (Signal-to-Distortion Ratio)** – use [`mir_eval`](https://github.com/craffel/mir_eval) or `museval`.
* **SI-SDR** (scale-invariant SDR) – more modern and fairer metric.

