# %%writefile parkinsons_voice_app.py
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
from audio_recorder_streamlit import audio_recorder  # NEW mic recorder

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide"
)

# Load model — no fallback
@st.cache_resource
def load_model():
    if not os.path.exists("parkinsons_model.pkl"):
        st.error("Model file 'parkinsons_model.pkl' not found. Please upload it.")
        st.stop()
    return joblib.load("parkinsons_model.pkl")

def safe_praat_call(func, *args, default=0):
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {str(e)}")
        return default

def create_pdf_report(features, prediction, proba, audio_plots):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Parkinson's Voice Analysis Report", ln=1, align='C')
    pdf.ln(10)
    
    # Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnostic Results", ln=1)
    pdf.set_font("Arial", size=12)
    
    pred_text = "Possible Parkinson's" if prediction else "Healthy"
    pdf.cell(200, 10, txt=f"Prediction: {pred_text}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {proba*100:.1f}%", ln=1)
    pdf.ln(10)
    
    # Features
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Feature Analysis", ln=1)
    pdf.set_font("Arial", size=10)
    
    col_width = pdf.w / 3
    row_height = pdf.font_size * 1.5
    for i, (k, v) in enumerate(features.items()):
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 240)
        else:
            pdf.set_fill_color(255, 255, 255)
            
        pdf.cell(col_width, row_height, txt=k, border=1, fill=True)
        pdf.cell(col_width, row_height, txt=f"{v:.4f}", border=1, fill=True, ln=1)
    
    for plot in audio_plots:
        pdf.add_page()
        pdf.image(plot, x=10, y=10, w=180)
        try:
            os.unlink(plot)
        except:
            pass
    
    return bytes(pdf.output(dest='S'))

def plot_audio_features(y, sr):
    plots = []
    
    # Waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    waveform_path = os.path.join(tempfile.gettempdir(), f"waveform_{int(time.time())}.png")
    plt.savefig(waveform_path, bbox_inches='tight')
    plt.close()
    plots.append(waveform_path)
    
    # Spectrogram plot
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join(tempfile.gettempdir(), f"spectrogram_{int(time.time())}.png")
    plt.savefig(spectrogram_path, bbox_inches='tight')
    plt.close()
    plots.append(spectrogram_path)
    
    return plots

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
            st.warning("Audio too short for analysis (minimum 0.5 seconds)")
            return None, None

        audio_plots = plot_audio_features(y, sr)

        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]

        if len(f0_values) > 10:
            features.update({
                "MDVP:Fo(Hz)": np.mean(f0_values),
                "MDVP:Fhi(Hz)": np.max(f0_values),
                "MDVP:Flo(Hz)": np.min(f0_values),
                "PPE": entropy(f0_values) if len(f0_values) > 1 else 0
            })

        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
        jitter_local = safe_praat_call(parselmouth.praat.call, point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = safe_praat_call(parselmouth.praat.call, [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        features.update({
            "MDVP:Jitter(%)": jitter_local,
            "MDVP:Jitter(Abs)": jitter_local,
            "MDVP:RAP": jitter_local,
            "MDVP:PPQ": jitter_local,
            "Jitter:DDP": jitter_local * 3,
            "MDVP:Shimmer": shimmer_local,
            "MDVP:Shimmer(dB)": shimmer_local,
            "Shimmer:APQ3": shimmer_local / 3,
            "Shimmer:APQ5": shimmer_local / 5,
            "MDVP:APQ": shimmer_local,
            "Shimmer:DDA": shimmer_local * 3,
        })

        harmonicity = snd.to_harmonicity_cc()
        hnr = safe_praat_call(parselmouth.praat.call, harmonicity, "Get mean", 0, 0)
        features["HNR"] = hnr
        features["NHR"] = 1 / hnr if hnr > 0 else 0

        features.update({
            "RPDE": entropy(f0_values),
            "DFA": librosa.feature.rms(y=y).mean(),
            "spread1": np.std(f0_values),
            "spread2": np.var(f0_values),
            "D2": np.percentile(f0_values, 99),
        })

        return features, audio_plots

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None, None
    finally:
        time.sleep(0.5)
        try:
            os.unlink(audio_path)
        except:
            pass

def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("Record or upload a short 'ahhh' recording (3-5 seconds) for analysis")

    with st.sidebar:
        st.header("Instructions")
        st.markdown("1. Record using microphone or upload WAV file\n2. Click Analyze\n3. View results and download report")

    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    st.subheader("🎙 Live Recording")
    audio_bytes = audio_recorder(pause_threshold=3.0, sample_rate=44100)
    if audio_bytes:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        st.session_state.audio_file = temp_audio_path
        st.audio(audio_bytes, format="audio/wav")

    st.subheader("📂 Or Upload Audio File")
    uploaded_file = st.file_uploader("Choose WAV file", type=["wav"])
    if uploaded_file:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.audio_file = temp_audio_path
        st.audio(uploaded_file, format="audio/wav")

    if st.button("Analyze Voice"):
        if not st.session_state.audio_file:
            st.warning("Please record or upload an audio file first.")
            return
        
        with st.spinner("Analyzing..."):
            features, audio_plots = extract_features(st.session_state.audio_file)
            if features:
                df = pd.DataFrame([features])
                model = load_model()
                prediction = model.predict(df)[0]
                proba = model.predict_proba(df)[0][1]

                st.subheader("Results")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", "🧠 Possible Parkinson's" if prediction else "✅ Healthy", f"{proba*100:.1f}%")
                col2.metric("Risk Level", "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low")

                with st.expander("Audio Visualizations"):
                    cols = st.columns(2)
                    cols[0].image(audio_plots[0], caption="Waveform")
                    cols[1].image(audio_plots[1], caption="Spectrogram")

                with st.expander("Feature Details"):
                    st.dataframe(df.T.style.background_gradient(cmap="Blues"))

                pdf_report = create_pdf_report(features, prediction, proba, audio_plots)
                st.download_button(
                    label="Download Report",
                    data=pdf_report,
                    file_name="parkinson_analysis.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
