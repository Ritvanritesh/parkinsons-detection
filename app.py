# parkinsons_voice_app.py
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
# Use Agg backend to avoid GUI issues on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import warnings
import soundfile as sf

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Parkinson's Voice Analysis", page_icon="🧠", layout="wide")
MODEL_PATH_CANDIDATE = "/mnt/data/parkinsons_model.pkl"  # path you uploaded to
MODEL_FALLBACK_NAME = "parkinsons_model.pkl"  # name if user uploads
MIN_DURATION_SEC = 0.5

# -------------------------
# Safe Praat helper
# -------------------------
def safe_praat_call(func, *args, default=0):
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {str(e)}")
        return default

# -------------------------
# Model loading: load from candidate path or sidebar upload
# -------------------------
@st.cache_resource
def load_model_from_path(path):
    return joblib.load(path)

def get_model():
    # try candidate path first
    if os.path.exists(MODEL_PATH_CANDIDATE):
        try:
            return load_model_from_path(MODEL_PATH_CANDIDATE)
        except Exception as e:
            st.sidebar.error(f"Failed to load model at {MODEL_PATH_CANDIDATE}: {e}")
    # else try app folder name
    if os.path.exists(MODEL_FALLBACK_NAME):
        try:
            return load_model_from_path(MODEL_FALLBACK_NAME)
        except Exception as e:
            st.sidebar.error(f"Failed to load model '{MODEL_FALLBACK_NAME}': {e}")
    # allow upload
    uploaded = st.sidebar.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib", "sav"])
    if uploaded:
        # save to fallback name and load
        with open(MODEL_FALLBACK_NAME, "wb") as f:
            f.write(uploaded.getvalue())
        try:
            return load_model_from_path(MODEL_FALLBACK_NAME)
        except Exception as e:
            st.sidebar.error(f"Uploaded model failed to load: {e}")
    # if nothing, show message and stop
    st.sidebar.info(f"Provide a trained model named '{os.path.basename(MODEL_PATH_CANDIDATE)}' in /mnt/data or upload one here.")
    return None

# -------------------------
# Plot helpers (avoid prop_cycler issues)
# -------------------------
def plot_audio_features(y, sr):
    plots = []
    # reset rcParams to defaults for safe plotting
    plt.rcParams.update(plt.rcParamsDefault)

    # Waveform: avoid relying on matplotlib style cycler side effects
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    waveform_path = os.path.join(tempfile.gettempdir(), f"waveform_{int(time.time()*1000)}.png")
    fig.savefig(waveform_path, bbox_inches='tight')
    plt.close(fig)
    plots.append(waveform_path)

    # Spectrogram using librosa (safe after rcParams reset)
    fig, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Spectrogram (log freq)")
    spectrogram_path = os.path.join(tempfile.gettempdir(), f"spectrogram_{int(time.time()*1000)}.png")
    fig.savefig(spectrogram_path, bbox_inches='tight')
    plt.close(fig)
    plots.append(spectrogram_path)

    return plots

# -------------------------
# Feature extraction (parselmouth + librosa)
# Returns (features_dict, plots_list)
# -------------------------
def extract_features(audio_path):
    features = {
        "MDVP:Fo(Hz)": 0, "MDVP:Fhi(Hz)": 0, "MDVP:Flo(Hz)": 0,
        "MDVP:Jitter(%)": 0, "MDVP:Jitter(Abs)": 0, "MDVP:RAP": 0,
        "MDVP:PPQ": 0, "Jitter:DDP": 0, "MDVP:Shimmer": 0,
        "MDVP:Shimmer(dB)": 0, "Shimmer:APQ3": 0, "Shimmer:APQ5": 0,
        "MDVP:APQ": 0, "Shimmer:DDA": 0, "NHR": 0, "HNR": 0,
        "RPDE": 0, "DFA": 0, "spread1": 0, "spread2": 0,
        "D2": 0, "PPE": 0
    }

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        snd = parselmouth.Sound(audio_path)

        if snd.duration < MIN_DURATION_SEC:
            st.warning(f"Audio too short for analysis (min {MIN_DURATION_SEC} s).")
            return None, None

        audio_plots = plot_audio_features(y, sr)

        # pitch and f0 values
        pitch = snd.to_pitch()
        try:
            f0_values = pitch.selected_array['frequency']
        except Exception:
            f0_values = np.array([])
        f0_values = f0_values[f0_values > 0]

        if len(f0_values) > 0:
            features["MDVP:Fo(Hz)"] = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
            features["MDVP:Fhi(Hz)"] = float(np.max(f0_values)) if len(f0_values) > 0 else 0.0
            features["MDVP:Flo(Hz)"] = float(np.min(f0_values)) if len(f0_values) > 0 else 0.0
            features["PPE"] = float(entropy(f0_values)) if len(f0_values) > 1 else 0.0

        # PointProcess for jitter/shimmer
        point_process = safe_praat_call(parselmouth.praat.call, snd, "To PointProcess (periodic, cc)", 75, 300)
        jitter_local = safe_praat_call(parselmouth.praat.call, point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = safe_praat_call(parselmouth.praat.call, [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # convert to floats safely
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
        features["NHR"] = 1.0 / hnr if (hnr and hnr > 0) else 0.0

        # derived features
        features.update({
            "RPDE": float(entropy(f0_values)) if len(f0_values) > 0 else 0.0,
            "DFA": float(librosa.feature.rms(y=y).mean()) if len(y) > 0 else 0.0,
            "spread1": float(np.std(f0_values)) if len(f0_values) > 0 else 0.0,
            "spread2": float(np.var(f0_values)) if len(f0_values) > 0 else 0.0,
            "D2": float(np.percentile(f0_values, 99)) if len(f0_values) > 0 else 0.0,
        })

        return features, audio_plots

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None, None
    finally:
        # try removing temp file if in tempdir
        try:
            if audio_path.startswith(tempfile.gettempdir()):
                os.unlink(audio_path)
        except Exception:
            pass

# -------------------------
# PDF report
# -------------------------
def create_pdf_report(features, prediction, proba, audio_plots):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Parkinson's Voice Analysis Report", ln=1, align='C')
    pdf.ln(8)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, "Diagnostic Results", ln=1)
    pdf.set_font("Arial", size=11)
    pred_text = "Possible Parkinson's" if prediction else "Healthy"
    pdf.cell(200, 8, f"Prediction: {pred_text}", ln=1)
    pdf.cell(200, 8, f"Confidence: {proba*100:.1f}%" if not np.isnan(proba) else "Confidence: N/A", ln=1)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, "Feature Analysis (selected)", ln=1)
    pdf.set_font("Arial", size=10)
    col_width = pdf.w / 2 - 20
    row_h = pdf.font_size * 1.4
    for i, (k, v) in enumerate(features.items()):
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_width, row_h, k, border=1, fill=True)
        pdf.cell(col_width, row_h, f"{v:.4f}", border=1, fill=True, ln=1)

    for plot in audio_plots:
        pdf.add_page()
        try:
            pdf.image(plot, x=10, y=10, w=190)
        except Exception:
            pass
        try:
            os.unlink(plot)
        except Exception:
            pass

    return pdf.output(dest='S').encode('latin-1')

# -------------------------
# Main UI
# -------------------------
def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("Record or upload a short sustained vowel (e.g., 'ahhh') for 3–5 seconds.")

    with st.sidebar:
        st.header("Model & Instructions")
        st.write(f"- Model file expected at `{MODEL_PATH_CANDIDATE}` or upload one below.")
        st.write("- Input: short sustained vowel (3–5 s), mono WAV recommended, 22050 Hz sample rate.")
        st.file_uploader("Upload model (.pkl/.joblib)", type=["pkl", "joblib", "sav"])

    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None

    st.subheader("Record (browser)")
    st.write("Click record and say a sustained vowel (~3–5s).")
    audio_bytes = audio_recorder(sample_rate=22050, pause_threshold=2.0, max_length=8)

    if audio_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        st.session_state.audio_path = tmp.name
        st.audio(audio_bytes, format="audio/wav")

    st.subheader("Or upload a file")
    uploaded = st.file_uploader("Upload WAV file", type=["wav", "mp3"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded.getvalue())
        tmp.flush()
        tmp.close()
        st.session_state.audio_path = tmp.name
        st.audio(uploaded.getvalue(), format="audio/wav")

    analyze = st.button("Analyze Voice")
    if analyze:
        if not st.session_state.audio_path:
            st.warning("Please record or upload an audio file first.")
            return

        with st.spinner("Extracting features and predicting..."):
            features, plots = extract_features(st.session_state.audio_path)
            if features is None:
                return

            model = get_model()
            if model is None:
                st.error("Model not available. Upload a trained model in the sidebar or place it at the expected path.")
                return

            df = pd.DataFrame([features])
            try:
                prediction = int(model.predict(df)[0])
            except Exception as e:
                st.error(f"Model failed to predict: {e}")
                return

            try:
                proba = float(model.predict_proba(df)[0][1])
            except Exception:
                proba = float("nan")

            st.success("Analysis complete!")
            col1, col2 = st.columns(2)
            col1.metric("Prediction", "🧠 Possible Parkinson's" if prediction else "✅ Healthy",
                        f"{proba*100:.1f}%" if not np.isnan(proba) else "N/A")
            col2.metric("Risk Level", "High" if (not np.isnan(proba) and proba > 0.7) else
                        "Medium" if (not np.isnan(proba) and proba > 0.5) else "Low")

            with st.expander("Audio Visualizations"):
                try:
                    st.image(plots[0], caption="Waveform")
                    st.image(plots[1], caption="Spectrogram")
                except Exception as e:
                    st.warning(f"Could not show plots: {e}")

            with st.expander("Feature values (technical)"):
                st.dataframe(df.T.style.background_gradient(cmap="Blues"))

            pdf_bytes = create_pdf_report(features, prediction, proba, plots)
            st.download_button("📄 Download PDF report", data=pdf_bytes, file_name="parkinson_analysis.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
