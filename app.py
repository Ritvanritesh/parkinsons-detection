# app.py
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import librosa
import librosa.display
import numpy as np
import parselmouth
import pandas as pd
from scipy.stats import entropy
import joblib
import os
import tempfile
import matplotlib
matplotlib.use("Agg")  # safe backend for servers
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import warnings
import soundfile as sf

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Parkinson's Voice Analysis", page_icon="🧠", layout="wide")
MODEL_FILENAME = "parkinsons_model.pkl"
MIN_DURATION = 0.5  # seconds

# -----------------------
# Helpers
# -----------------------
def safe_praat_call(func, *args, default=0):
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {e}")
        return default

def reset_plot_defaults():
    plt.rcParams.update(plt.rcParamsDefault)

# -----------------------
# Model loading (from disk or upload)
# -----------------------
@st.cache_resource
def load_model_from_disk(path):
    return joblib.load(path)

def get_model():
    # try common path first
    if os.path.exists(MODEL_FILENAME):
        try:
            return load_model_from_disk(MODEL_FILENAME)
        except Exception as e:
            st.sidebar.error(f"Failed to load model '{MODEL_FILENAME}': {e}")

    uploaded = st.sidebar.file_uploader("Upload trained model (.pkl/.joblib)", type=["pkl", "joblib"])
    if uploaded:
        temp_model_path = MODEL_FILENAME
        with open(temp_model_path, "wb") as f:
            f.write(uploaded.getvalue())
        try:
            return load_model_from_disk(temp_model_path)
        except Exception as e:
            st.sidebar.error(f"Uploaded model failed to load: {e}")
            return None

    st.sidebar.info(f"Place '{MODEL_FILENAME}' into app folder or upload it here.")
    return None

# -----------------------
# Plot creation (safe)
# -----------------------
def plot_audio_features(y, sr):
    plots = []
    # reset rc params to avoid prop_cycler issues
    reset_plot_defaults()

    # waveform (use plain matplotlib to avoid display cycler issues)
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    waveform_path = os.path.join(tempfile.gettempdir(), f"waveform_{int(time.time()*1000)}.png")
    fig.savefig(waveform_path, bbox_inches="tight")
    plt.close(fig)
    plots.append(waveform_path)

    # spectrogram using librosa.specshow
    fig, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram (log freq)")
    spectrogram_path = os.path.join(tempfile.gettempdir(), f"spectrogram_{int(time.time()*1000)}.png")
    fig.savefig(spectrogram_path, bbox_inches="tight")
    plt.close(fig)
    plots.append(spectrogram_path)

    return plots

# -----------------------
# Feature extraction (22 features)
# Returns (features_dict, list_of_plot_paths)
# -----------------------
def extract_features(audio_path):
    features = {
        "MDVP:Fo(Hz)": 0.0, "MDVP:Fhi(Hz)": 0.0, "MDVP:Flo(Hz)": 0.0,
        "MDVP:Jitter(%)": 0.0, "MDVP:Jitter(Abs)": 0.0, "MDVP:RAP": 0.0,
        "MDVP:PPQ": 0.0, "Jitter:DDP": 0.0, "MDVP:Shimmer": 0.0,
        "MDVP:Shimmer(dB)": 0.0, "Shimmer:APQ3": 0.0, "Shimmer:APQ5": 0.0,
        "MDVP:APQ": 0.0, "Shimmer:DDA": 0.0, "NHR": 0.0, "HNR": 0.0,
        "RPDE": 0.0, "DFA": 0.0, "spread1": 0.0, "spread2": 0.0,
        "D2": 0.0, "PPE": 0.0
    }

    try:
        # load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        snd = parselmouth.Sound(audio_path)

        if snd.duration < MIN_DURATION:
            st.warning(f"Audio too short (min {MIN_DURATION}s).")
            return None, None

        plots = plot_audio_features(y, sr)

        # pitch
        pitch = snd.to_pitch()
        try:
            f0 = pitch.selected_array['frequency']
        except Exception:
            f0 = np.array([])
        f0 = f0[f0 > 0]

        if len(f0) > 0:
            features["MDVP:Fo(Hz)"] = float(np.mean(f0))
            features["MDVP:Fhi(Hz)"] = float(np.max(f0))
            features["MDVP:Flo(Hz)"] = float(np.min(f0))
            features["PPE"] = float(entropy(f0)) if len(f0) > 1 else 0.0

        # point process + jitter/shimmer with safe praat calls
        point_process = safe_praat_call(parselmouth.praat.call, snd, "To PointProcess (periodic, cc)", 75, 300)
        jitter_local = safe_praat_call(parselmouth.praat.call, point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = safe_praat_call(parselmouth.praat.call, [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        try:
            jitter_local = float(jitter_local)
        except Exception:
            jitter_local = 0.0
        try:
            shimmer_local = float(shimmer_local)
        except Exception:
            shimmer_local = 0.0

        features.update({
            "MDVP:Jitter(%)": jitter_local,
            "MDVP:Jitter(Abs)": jitter_local,
            "MDVP:RAP": jitter_local,
            "MDVP:PPQ": jitter_local,
            "Jitter:DDP": jitter_local * 3 if jitter_local else 0.0,
            "MDVP:Shimmer": shimmer_local,
            "MDVP:Shimmer(dB)": shimmer_local,
            "Shimmer:APQ3": shimmer_local / 3 if shimmer_local else 0.0,
            "Shimmer:APQ5": shimmer_local / 5 if shimmer_local else 0.0,
            "MDVP:APQ": shimmer_local,
            "Shimmer:DDA": shimmer_local * 3 if shimmer_local else 0.0,
        })

        # harmonicity
        harmonicity = snd.to_harmonicity_cc()
        hnr = safe_praat_call(parselmouth.praat.call, harmonicity, "Get mean", 0, 0)
        try:
            hnr = float(hnr)
        except Exception:
            hnr = 0.0
        features["HNR"] = hnr
        features["NHR"] = 1.0 / hnr if hnr and hnr > 0 else 0.0

        # derived features
        features.update({
            "RPDE": float(entropy(f0)) if len(f0) > 0 else 0.0,
            "DFA": float(librosa.feature.rms(y=y).mean()) if len(y) > 0 else 0.0,
            "spread1": float(np.std(f0)) if len(f0) > 0 else 0.0,
            "spread2": float(np.var(f0)) if len(f0) > 0 else 0.0,
            "D2": float(np.percentile(f0, 99)) if len(f0) > 0 else 0.0,
        })

        return features, plots

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None, None

# -----------------------
# PDF report builder
# -----------------------
def create_pdf_report(features: dict, prediction: int, proba: float, plots: list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Parkinson's Voice Analysis Report", ln=1, align='C')
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, "Diagnostic Results", ln=1)
    pdf.set_font("Arial", size=11)
    label = "Possible Parkinson's" if prediction else "Healthy"
    pdf.cell(200, 8, f"Prediction: {label}", ln=1)
    if not np.isnan(proba):
        pdf.cell(200, 8, f"Confidence: {proba*100:.1f}%", ln=1)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, "Features (selected)", ln=1)
    pdf.set_font("Arial", size=10)
    w = pdf.w / 2 - 20
    rh = pdf.font_size * 1.4
    for i, (k, v) in enumerate(features.items()):
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(w, rh, k, border=1, fill=True)
        pdf.cell(w, rh, f"{v:.4f}", border=1, fill=True, ln=1)

    for p in plots:
        pdf.add_page()
        try:
            pdf.image(p, x=10, y=10, w=190)
        except Exception:
            pass

    return pdf.output(dest='S').encode('latin-1')

# -----------------------
# UI
# -----------------------
def main():
    st.title("🧠 Parkinson's Voice Analysis")
    st.markdown("Upload a WAV file or **record in-browser** (press Record) and say a sustained vowel for ~3–5s.")

    with st.sidebar:
        st.header("Model & Instructions")
        st.write(f"- Add your trained model file named **{MODEL_FILENAME}** in the app folder or upload below.")
        st.write("- Input recommended: sustained vowel (3–5s), mono WAV, 22050 Hz.")
        st.file_uploader("Upload model (.pkl/.joblib)", type=["pkl", "joblib"])

    model = get_model()  # may be None

    # session state for audio path
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None

    # Recorder
    st.subheader("Record (in-browser)")
    st.write("Click to record, then stop when done. Record ~3–5 seconds.")
    audio_bytes = audio_recorder(sample_rate=22050, pause_threshold=2.0, max_length=8)

    if audio_bytes:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmpf.write(audio_bytes)
        tmpf.flush()
        tmpf.close()
        st.session_state.audio_path = tmpf.name
        st.audio(audio_bytes, format="audio/wav")

    # Upload fallback
    st.subheader("Or upload a WAV file")
    uploaded = st.file_uploader("Upload WAV", type=["wav"])
    if uploaded is not None:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmpf.write(uploaded.getvalue())
        tmpf.flush()
        tmpf.close()
        st.session_state.audio_path = tmpf.name
        st.audio(uploaded.getvalue(), format="audio/wav")

    # Analyze button
    if st.button("Analyze Voice"):
        if not st.session_state.audio_path:
            st.warning("Please record or upload an audio file first.")
            return

        with st.spinner("Extracting features and predicting..."):
            features, plots = extract_features(st.session_state.audio_path)
            if features is None:
                return

            # Create DataFrame with exact column order expected by your model
            col_order = list(features.keys())
            df = pd.DataFrame([features])[col_order]

            if model is None:
                st.error("Model not loaded. Upload a trained model in the sidebar or place it in the app folder.")
                return

            try:
                prediction = int(model.predict(df)[0])
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                return

            try:
                proba = float(model.predict_proba(df)[0][1])
            except Exception:
                proba = float("nan")

            st.success("Analysis complete")
            c1, c2 = st.columns(2)
            c1.metric("Prediction", "🧠 Possible Parkinson's" if prediction else "✅ Healthy",
                      f"{proba*100:.1f}%" if not np.isnan(proba) else "N/A")
            c2.metric("Risk Level", "High" if (not np.isnan(proba) and proba > 0.7) else
                      "Medium" if (not np.isnan(proba) and proba > 0.5) else "Low")

            with st.expander("Audio Visualizations"):
                for p in plots:
                    st.image(p)

            with st.expander("Feature values (technical)"):
                st.dataframe(df.T.style.background_gradient(cmap="Blues"))

            pdf_bytes = create_pdf_report(features, prediction, proba, plots)
            st.download_button("📄 Download PDF report", data=pdf_bytes, file_name="parkinson_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
