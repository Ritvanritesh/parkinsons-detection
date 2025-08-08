# parkinsons_voice_app.py
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import librosa
import numpy as np
import parselmouth
import pandas as pd
from scipy.stats import entropy
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import warnings

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide"
)

# --- 1. Helper Functions ---
def safe_praat_call(func, *args, default=0):
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {str(e)}")
        return default

def plot_audio_features(y, sr):
    """Generate waveform and spectrogram plots"""
    plots = []
    
    # Waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_path = os.path.join(tempfile.gettempdir(), f"waveform_{time.time()}.png")
    plt.savefig(waveform_path, bbox_inches='tight')
    plt.close()
    plots.append(waveform_path)
    
    # Spectrogram plot
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join(tempfile.gettempdir(), f"spectrogram_{time.time()}.png")
    plt.savefig(spectrogram_path, bbox_inches='tight')
    plt.close()
    plots.append(spectrogram_path)
    
    return plots

# --- 2. Model Loading ---
@st.cache_resource
def load_model():
    if not os.path.exists("parkinsons_model.pkl"):
        st.error("❌ Model file 'parkinsons_model.pkl' not found. Please add it to your app directory.")
        st.stop()
    return joblib.load("parkinsons_model.pkl")

# --- 3. Feature Extraction ---
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

        # Pitch analysis
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

        # Jitter and shimmer analysis
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

        # Harmonicity analysis
        harmonicity = snd.to_harmonicity_cc()
        hnr = safe_praat_call(parselmouth.praat.call, harmonicity, "Get mean", 0, 0)
        features["HNR"] = hnr
        features["NHR"] = 1 / hnr if hnr > 0 else 0

        # Nonlinear features
        features.update({
            "RPDE": entropy(f0_values),
            "DFA": librosa.feature.rms(y=y).mean(),
            "spread1": np.std(f0_values),
            "spread2": np.var(f0_values),
            "D2": np.percentile(f0_values, 99),
        })

        return features, plot_audio_features(y, sr)

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None, None
    finally:
        try:
            os.unlink(audio_path)
        except:
            pass

# --- 4. PDF Report Generation ---
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
    
    # Add plots
    for plot in audio_plots:
        pdf.add_page()
        pdf.image(plot, x=10, y=10, w=180)
        try:
            os.unlink(plot)
        except:
            pass
    
    return pdf.output(dest='S').encode('latin-1')

# --- 5. Main Application ---
def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("Record or upload a short 'ahhh' recording (3-5 seconds) for analysis")

    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Record using microphone or upload WAV file
        2. Click Analyze
        3. View results and download report
        """)

    # Audio input section
    audio_source = None
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Recording")
        audio_bytes = audio_recorder(
            text="Click to record (say 'ahhh' for 3-5 seconds):",
            pause_threshold=5.0,
            recording_color="#e8b62c",
            neutral_color="#6aa36f"
        )
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_source = tmp.name
            st.audio(audio_bytes, format="audio/wav")

    with col2:
        st.subheader("Or Upload Audio File")
        uploaded_file = st.file_uploader("Choose WAV file", type=["wav"], label_visibility="collapsed")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_source = tmp.name
            st.audio(uploaded_file.getvalue(), format="audio/wav")

    # Analysis section
    if st.button("Analyze Voice", type="primary") and audio_source:
        with st.spinner("Analyzing voice patterns..."):
            features, audio_plots = extract_features(audio_source)
            
            if features:
                model = load_model()
                df = pd.DataFrame([features])
                
                try:
                    prediction = model.predict(df)[0]
                    proba = model.predict_proba(df)[0][1]
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Prediction", 
                        "🧠 Possible Parkinson's" if prediction else "✅ Healthy",
                        f"{proba*100:.1f}%"
                    )
                    col2.metric(
                        "Confidence Level", 
                        "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low"
                    )
                    
                    # Visualizations
                    with st.expander("Audio Analysis"):
                        st.image(audio_plots[0], caption="Waveform")
                        st.image(audio_plots[1], caption="Spectrogram")
                    
                    # Feature details
                    with st.expander("Technical Features"):
                        st.dataframe(df.T.style.background_gradient(cmap="Blues"))
                    
                    # Download report
                    pdf_report = create_pdf_report(features, prediction, proba, audio_plots)
                    st.download_button(
                        label="📄 Download Full Report",
                        data=pdf_report,
                        file_name="parkinson_analysis.pdf",
                        mime="application/pdf"
                    )
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
