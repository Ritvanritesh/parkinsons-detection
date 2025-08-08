# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import os
import joblib
import pandas as pd
from fpdf import FPDF
import time
from scipy import signal
from scipy.stats import entropy

st.set_page_config(page_title="Parkinson's Voice Analysis", layout="wide", page_icon="🧠")

MODEL_FILENAME = "parkinsons_model.pkl"
MIN_DURATION = 0.5  # seconds

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_model_from_disk(path):
    return joblib.load(path)

def get_model():
    # 1) try file in folder
    if os.path.exists(MODEL_FILENAME):
        try:
            return load_model_from_disk(MODEL_FILENAME)
        except Exception as e:
            st.sidebar.error(f"Failed to load model '{MODEL_FILENAME}': {e}")

    # 2) ask user to upload
    uploaded = st.sidebar.file_uploader("Upload trained model (.pkl/.joblib)", type=["pkl", "joblib"])
    if uploaded:
        with open(MODEL_FILENAME, "wb") as f:
            f.write(uploaded.getvalue())
        try:
            return load_model_from_disk(MODEL_FILENAME)
        except Exception as e:
            st.sidebar.error(f"Uploaded model failed to load: {e}")
            return None

    st.sidebar.info(f"Place '{MODEL_FILENAME}' into app folder or upload one.")
    return None

# -------------------------
# Audio -> feature helpers
# -------------------------
def save_audio_bytes_to_wav(audio_bytes: bytes) -> str:
    """Save bytes to temporary wav file (soundfile compatible)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

def normalize_audio(y):
    if np.max(np.abs(y)) > 0:
        return y / np.max(np.abs(y))
    return y

def approximate_jitter_shimmer(y, sr):
    """
    Approximate jitter (freq perturbation) and shimmer (amp perturbation) without Praat.
    Uses frame-wise pitch (YIN) and short-time energy.
    Returns (jitter_local, shimmer_local).
    """
    # estimate f0 across frames using YIN
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=256)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
    except Exception:
        f0 = np.array([])

    if f0.size > 1:
        jitter_local = np.mean(np.abs(np.diff(f0))) / (np.mean(f0) + 1e-8)
    else:
        jitter_local = 0.0

    # shimmer approximate: variability of short-term RMS
    try:
        hop_length = 256
        frame_length = 1024
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        if rms.size > 1:
            shimmer_local = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
        else:
            shimmer_local = 0.0
    except Exception:
        shimmer_local = 0.0

    return float(jitter_local), float(shimmer_local)

def approx_hnr(y, sr):
    """
    Approximate harmonic-to-noise ratio by separating harmonic component using HPSS.
    """
    try:
        harm, percuss = librosa.effects.hpss(y)
        noise = y - harm
        h_power = np.mean(harm**2) + 1e-10
        n_power = np.mean(noise**2) + 1e-10
        return float(10 * np.log10(h_power / n_power))
    except Exception:
        return 0.0

def extract_22_features(audio_path):
    """
    Return features_dict (22 features) computed from audio_path.
    Order matches keys() of the returned dict.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y = normalize_audio(y)

    duration = len(y) / sr
    if duration < MIN_DURATION:
        raise ValueError(f"Audio too short ({duration:.2f}s). Minimum {MIN_DURATION}s required.")

    features = {}

    # Fundamental frequency stats (YIN)
    try:
        f0_frames = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=256)
        f0_vals = f0_frames[np.isfinite(f0_frames) & (f0_frames > 0)]
    except Exception:
        f0_vals = np.array([])

    features["MDVP:Fo(Hz)"] = float(np.mean(f0_vals)) if f0_vals.size > 0 else 0.0
    features["MDVP:Fhi(Hz)"] = float(np.max(f0_vals)) if f0_vals.size > 0 else 0.0
    features["MDVP:Flo(Hz)"] = float(np.min(f0_vals)) if f0_vals.size > 0 else 0.0

    # jitter & shimmer approximations
    jitter_local, shimmer_local = approximate_jitter_shimmer(y, sr)
    features["MDVP:Jitter(%)"] = jitter_local
    features["MDVP:Jitter(Abs)"] = jitter_local
    features["MDVP:RAP"] = jitter_local
    features["MDVP:PPQ"] = jitter_local
    features["Jitter:DDP"] = jitter_local * 3.0

    features["MDVP:Shimmer"] = shimmer_local
    features["MDVP:Shimmer(dB)"] = 20.0 * np.log10(shimmer_local + 1e-8) if shimmer_local > 0 else 0.0
    features["Shimmer:APQ3"] = shimmer_local / 3.0
    features["Shimmer:APQ5"] = shimmer_local / 5.0
    features["MDVP:APQ"] = shimmer_local
    features["Shimmer:DDA"] = shimmer_local * 3.0

    # HNR approx and NHR
    hnr = approx_hnr(y, sr)
    features["HNR"] = hnr
    features["NHR"] = 1.0 / (hnr + 1e-8) if hnr != 0 else 0.0

    # Nonlinear / derived features
    # RPDE approx: use spectral entropy of pitch distribution
    if f0_vals.size > 1:
        # discretize and compute entropy
        hist, _ = np.histogram(f0_vals, bins=20, density=True)
        hist = hist + 1e-8
        features["RPDE"] = float(entropy(hist))
        features["PPE"] = float(entropy(f0_vals))
    else:
        features["RPDE"] = 0.0
        features["PPE"] = 0.0

    # DFA approx: use RMS mean as proxy
    features["DFA"] = float(librosa.feature.rms(y=y).mean())

    # spread1 & spread2 (statistical spreads)
    features["spread1"] = float(np.std(f0_vals)) if f0_vals.size > 0 else 0.0
    features["spread2"] = float(np.var(f0_vals)) if f0_vals.size > 0 else 0.0
    features["D2"] = float(np.percentile(f0_vals, 99)) if f0_vals.size > 0 else 0.0

    # If we still have fewer than 22 keys, add more spectral features to reach 22
    # Current keys count
    # Ensure exactly 22 features
    # Add spectral centroid, bandwidth, rolloff, zero-crossing, mfcc means to fill slots
    extra = {}
    extra["spectral_centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    extra["spectral_bandwidth"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    extra["spectral_rolloff"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    extra["zero_crossing_rate"] = float(librosa.feature.zero_crossing_rate(y).mean())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i in range(mfcc.shape[0]):
        extra[f"mfcc_mean_{i+1}"] = float(mfcc[i].mean())

    # Merge features and extras, but respect "22 features" count:
    all_features = {**features, **extra}
    # If more than 22, truncate by insertion order
    # If fewer, pad with zeros
    keys = list(all_features.keys())
    if len(keys) >= 22:
        keys = keys[:22]
    else:
        # add zero-padding keys
        i = 0
        while len(keys) < 22:
            keys.append(f"pad_{i}")
            all_features[f"pad_{i}"] = 0.0
            i += 1

    result = {k: float(all_features[k]) for k in keys}
    return result

# -------------------------
# Plotting helpers
# -------------------------
def create_waveform_plot(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    t = np.linspace(0, len(y)/sr, num=len(y))
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def create_spectrogram_plot(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram (log)")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def create_feature_barplot(features_dict):
    keys = list(features_dict.keys())
    vals = [features_dict[k] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_title("Feature magnitudes (truncated 22)")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# -------------------------
# PDF report builder
# -------------------------
def create_pdf(features_dict, prediction_label, proba, plot_paths):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Parkinson's Voice Analysis Report", ln=1, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, f"Prediction: {prediction_label}", ln=1)
    if proba is not None:
        try:
            pdf.cell(200, 8, f"Confidence: {proba*100:.1f}%", ln=1)
        except Exception:
            pdf.cell(200, 8, f"Confidence: N/A", ln=1)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, "Features (truncated)", ln=1)
    pdf.set_font("Arial", size=10)
    col_w = pdf.w / 2 - 20
    row_h = pdf.font_size * 1.4
    i = 0
    for k, v in features_dict.items():
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_w, row_h, k, border=1, fill=True)
        pdf.cell(col_w, row_h, f"{v:.4f}", border=1, fill=True, ln=1)
        i += 1

    for p in plot_paths:
        pdf.add_page()
        try:
            pdf.image(p, x=10, y=20, w=190)
        except Exception:
            pass

    return pdf.output(dest="S").encode("latin-1")

# -------------------------
# App layout & logic
# -------------------------
def main():
    st.title("🧠 Parkinson's Voice Analysis (Streamlit-webrtc, upload & record)")

    st.sidebar.header("Model")
    model = get_model()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Record in-browser (webrtc)")
        st.write("Press the Start button below, then speak a sustained vowel for ~3–5s, then press Stop.")
        webrtc_ctx = webrtc_streamer(key="audio", mode=WebRtcMode.SENDONLY, media_stream_constraints={"audio": True, "video": False})
        record_btn = st.button("Capture last few seconds (save to file)")

        if webrtc_ctx.state.playing and record_btn:
            # pull frames from audio receiver if available
            audio_receiver = webrtc_ctx.audio_receiver
            if audio_receiver is None:
                st.warning("Audio receiver not ready yet.")
            else:
                frames = audio_receiver.get_frames(timeout=1.0)
                if len(frames) == 0:
                    st.warning("No audio frames received.")
                else:
                    # concatenate frames into numpy array
                    all_samples = np.concatenate([f.to_ndarray()[:, 0] for f in frames])
                    sr = frames[0].sample_rate
                    # normalize and save to wav
                    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(tmp_wav.name, all_samples.astype(np.float32), sr)
                    st.success("Captured audio saved.")
                    st.session_state["latest_audio_path"] = tmp_wav.name

    with col2:
        st.subheader("Or upload a WAV file")
        uploaded = st.file_uploader("Upload WAV", type=["wav"])
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(uploaded.getvalue())
            tmp.flush()
            tmp.close()
            st.session_state["latest_audio_path"] = tmp.name
            st.audio(uploaded.getvalue())

    st.markdown("---")
    if "latest_audio_path" in st.session_state and st.session_state["latest_audio_path"]:
        audio_path = st.session_state["latest_audio_path"]
        st.write(f"Current audio: `{audio_path}`")
        if st.button("Analyze audio"):
            try:
                features = extract_22_features(audio_path)
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
                return

            # create dataframe in the expected column order
            df = pd.DataFrame([features])
            st.subheader("Extracted Features (truncated to 22)")
            st.dataframe(df.T.style.background_gradient(cmap="Blues"))

            if model is None:
                st.warning("No model loaded. Upload model in sidebar.")
            else:
                # ensure correct shape
                try:
                    # some models expect same column names; so we pass df directly
                    pred = model.predict(df)
                    if hasattr(model, "predict_proba"):
                        proba = float(model.predict_proba(df)[0][1])
                    else:
                        proba = None
                    label = "Possible Parkinson's" if int(pred[0]) == 1 else "Healthy"
                    st.metric("Prediction", label, f"{proba*100:.1f}%" if proba is not None else "")
                except Exception as e:
                    st.error(f"Model prediction error: {e}")
                    return

            # plots
            y, sr = librosa.load(audio_path, sr=22050)
            waveform_png = create_waveform_plot(y, sr)
            spectrogram_png = create_spectrogram_plot(y, sr)
            bar_png = create_feature_barplot(features)

            st.subheader("Visuals")
            st.image(waveform_png, caption="Waveform")
            st.image(spectrogram_png, caption="Spectrogram")
            st.image(bar_png, caption="Feature magnitudes")

            # PDF
            pdf_bytes = create_pdf(features, label, proba, [waveform_png, spectrogram_png, bar_png])
            st.download_button("Download PDF report", data=pdf_bytes, file_name="parkinson_report.pdf", mime="application/pdf")

    # cleanup: optional button to clear last audio
    if st.button("Clear last audio"):
        if "latest_audio_path" in st.session_state and st.session_state["latest_audio_path"]:
            try:
                os.unlink(st.session_state["latest_audio_path"])
            except Exception:
                pass
            st.session_state["latest_audio_path"] = None
            st.success("Cleared.")

if __name__ == "__main__":
    main()
