# parkinsons_voice_app.py
import streamlit as st
import librosa
import librosa.display
import numpy as np
import parselmouth
import pandas as pd
from scipy.stats import entropy
import joblib
import warnings
from io import BytesIO
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
from audio_recorder_streamlit import audio_recorder  # microphone recorder component
import soundfile as sf

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Parkinson's Voice Analysis", page_icon="🧠", layout="wide")
MODEL_PATH = "parkinsons_model.pklb"  # expected model filename

# -------------------------
# Helper: safe praat call
# -------------------------
def safe_praat_call(func, *args, default=0):
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {str(e)}")
        return default

# -------------------------
# Load model (no fallback)
# If model file missing, allow upload in sidebar
# -------------------------
@st.cache_resource
def load_model_from_disk(path):
    return joblib.load(path)

def get_model():
    # If file present, load it
    if os.path.exists(MODEL_PATH):
        try:
            return load_model_from_disk(MODEL_PATH)
        except Exception as e:
            st.sidebar.error(f"Failed to load existing model '{MODEL_PATH}': {e}")
            st.stop()
    # otherwise check sidebar uploader
    uploaded_model = st.sidebar.file_uploader("Upload model (.joblib or .pkl)", type=["joblib", "pkl"])
    if uploaded_model:
        try:
            # save uploaded model to disk
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.getvalue())
            return load_model_from_disk(MODEL_PATH)
        except Exception as e:
            st.sidebar.error(f"Failed to save/load uploaded model: {e}")
            st.stop()
    else:
        st.sidebar.info(f"Place your trained model file named '{MODEL_PATH}' in app folder or upload it here.")
        st.stop()

# -------------------------
# PDF report creator
# -------------------------
def create_pdf_report(features: dict, prediction: int, proba: float, audio_plots: list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Parkinson's Voice Analysis Report", ln=1, align='C')
    pdf.ln(8)

    # Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnostic Results", ln=1)
    pdf.set_font("Arial", size=12)
    pred_text = "Possible Parkinson's" if prediction else "Healthy"
    pdf.cell(200, 8, txt=f"Prediction: {pred_text}", ln=1)
    pdf.cell(200, 8, txt=f"Confidence: {proba*100:.1f}%", ln=1)
    pdf.ln(6)

    # Features
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 8, txt="Feature Analysis (selected values)", ln=1)
    pdf.set_font("Arial", size=10)
    col_width = pdf.w / 2 - 20
    row_height = pdf.font_size * 1.6
    i = 0
    for k, v in features.items():
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_width, row_height, txt=str(k), border=1, fill=True)
        pdf.cell(col_width, row_height, txt=f"{v:.4f}", border=1, fill=True, ln=1)
        i += 1

    # Add plots
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

    return bytes(pdf.output(dest='S'))

# -------------------------
# Plotting helpers
# -------------------------
def plot_audio_features(y, sr):
    plots = []

    # Waveform
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.tight_layout()
    waveform_path = os.path.join(tempfile.gettempdir(), f"waveform_{int(time.time()*1000)}.png")
    plt.savefig(waveform_path, bbox_inches='tight')
    plt.close()
    plots.append(waveform_path)

    # Spectrogram
    plt.figure(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (log scale)')
    plt.tight_layout()
    spectrogram_path = os.path.join(tempfile.gettempdir(), f"spectrogram_{int(time.time()*1000)}.png")
    plt.savefig(spectrogram_path, bbox_inches='tight')
    plt.close()
    plots.append(spectrogram_path)

    return plots

# -------------------------
# Feature extraction (parselmouth + librosa)
# Returns (features_dict, audio_plots)
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
        y, sr = librosa.load(audio_path, sr=22050)
        snd = parselmouth.Sound(audio_path)

        if snd.duration < 0.5:
            st.warning("Audio too short for analysis (minimum 0.5 seconds).")
            return None, None

        audio_plots = plot_audio_features(y, sr)

        # Pitch features
        pitch = snd.to_pitch()
        try:
            f0_values = pitch.selected_array['frequency']
        except Exception:
            f0_values = np.array([])
        f0_values = f0_values[f0_values > 0]

        if len(f0_values) > 0:
            features["MDVP:Fo(Hz)"] = float(np.mean(f0_values)) if len(f0_values) > 0 else 0
            features["MDVP:Fhi(Hz)"] = float(np.max(f0_values)) if len(f0_values) > 0 else 0
            features["MDVP:Flo(Hz)"] = float(np.min(f0_values)) if len(f0_values) > 0 else 0
            features["PPE"] = float(entropy(f0_values)) if len(f0_values) > 1 else 0

        # Point process for jitter/shimmer
        point_process = safe_praat_call(parselmouth.praat.call, snd, "To PointProcess (periodic, cc)", 75, 300)
        jitter_local = safe_praat_call(parselmouth.praat.call, point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = safe_praat_call(parselmouth.praat.call, [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        features.update({
            "MDVP:Jitter(%)": float(jitter_local),
            "MDVP:Jitter(Abs)": float(jitter_local),
            "MDVP:RAP": float(jitter_local),
            "MDVP:PPQ": float(jitter_local),
            "Jitter:DDP": float(jitter_local) * 3,
            "MDVP:Shimmer": float(shimmer_local),
            "MDVP:Shimmer(dB)": float(shimmer_local),
            "Shimmer:APQ3": float(shimmer_local) / 3 if shimmer_local else 0,
            "Shimmer:APQ5": float(shimmer_local) / 5 if shimmer_local else 0,
            "MDVP:APQ": float(shimmer_local),
            "Shimmer:DDA": float(shimmer_local) * 3 if shimmer_local else 0,
        })

        harmonicity = snd.to_harmonicity_cc()
        hnr = safe_praat_call(parselmouth.praat.call, harmonicity, "Get mean", 0, 0)
        features["HNR"] = float(hnr)
        features["NHR"] = float(1 / hnr) if (hnr and hnr > 0) else 0

        # Additional derived features
        features.update({
            "RPDE": float(entropy(f0_values)) if len(f0_values) > 0 else 0,
            "DFA": float(librosa.feature.rms(y=y).mean()),
            "spread1": float(np.std(f0_values)) if len(f0_values) > 0 else 0,
            "spread2": float(np.var(f0_values)) if len(f0_values) > 0 else 0,
            "D2": float(np.percentile(f0_values, 99)) if len(f0_values) > 0 else 0,
        })

        return features, audio_plots

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None, None
    finally:
        # remove temporary audio file if it's in temp folder
        try:
            if audio_path.startswith(tempfile.gettempdir()):
                os.unlink(audio_path)
        except Exception:
            pass

# -------------------------
# Main app UI
# -------------------------
def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("Record or upload a short sustained vowel (e.g., 'ahhh') for 3–5 seconds for analysis.")

    # Sidebar: model upload / instructions
    st.sidebar.header("Model & Instructions")
    st.sidebar.markdown(
        "- Provide a trained classifier file named `parkinsons_model.pklb` (or upload it here).\n"
        "- Recommended input: short sustained vowel (3–5s), mono WAV, 22050 Hz.\n"
        "- If the model is not present, upload it in the sidebar."
    )
    # Allow optional model upload in sidebar (handled in get_model)
    _ = st.sidebar.empty()

    # Session state for audio path
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    # Live recorder
    st.subheader("🎙 Live Recording (in-browser)")
    st.write("Press record, speak the sustained vowel for ~3–5s, then stop.")
    audio_bytes = audio_recorder(sample_rate=22050, pause_threshold=2.0, max_length=7)

    if audio_bytes:
        # audio_bytes is raw bytes of a WAV
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        st.session_state.audio_file = tmp.name
        st.audio(audio_bytes, format="audio/wav")

    # Or upload
    st.subheader("📂 Or Upload Audio File")
    uploaded_file = st.file_uploader("Choose WAV file (mono recommended)", type=["wav", "mp3"])
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        tmp.close()
        st.session_state.audio_file = tmp.name
        st.audio(uploaded_file)

    # Analyze button
    if st.button("Analyze Voice"):
        if not st.session_state.audio_file:
            st.warning("Please record or upload an audio file first.")
            return

        with st.spinner("Extracting features and predicting..."):
            features, audio_plots = extract_features(st.session_state.audio_file)
            if not features:
                return

            df = pd.DataFrame([features])

            # load model (or ask to upload)
            model = get_model()
            try:
                prediction = int(model.predict(df)[0])
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                return

            try:
                proba = float(model.predict_proba(df)[0][1])
            except Exception:
                # if predict_proba not available, set a NaN
                proba = float(np.nan)

            # Results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            pred_label = "🧠 Possible Parkinson's" if prediction else "✅ Healthy"
            conf_text = f"{proba*100:.1f}%" if not np.isnan(proba) else "N/A"
            col1.metric("Prediction", pred_label, conf_text)
            col2.metric("Risk Level", "High" if (not np.isnan(proba) and proba > 0.7) else "Medium" if (not np.isnan(proba) and proba > 0.5) else "Low")

            # Visualizations
            with st.expander("Audio Visualizations"):
                try:
                    cols = st.columns(2)
                    cols[0].image(audio_plots[0], caption="Waveform")
                    cols[1].image(audio_plots[1], caption="Spectrogram")
                except Exception as e:
                    st.warning(f"Could not show visualizations: {e}")

            with st.expander("Feature Details"):
                # show transposed DataFrame to be easy to read
                st.dataframe(df.T)

            # PDF report
            pdf_bytes = create_pdf_report(features, prediction, proba if not np.isnan(proba) else 0.0, audio_plots)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="parkinson_analysis.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
