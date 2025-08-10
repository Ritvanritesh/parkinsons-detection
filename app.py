# parkinsons_voice_app.py
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy
import joblib
import warnings
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
from datetime import datetime
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import wave
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load("parkinsons_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'parkinsons_model.pkl' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def save_audio_bytes(audio_bytes):
    """Save raw audio bytes to a WAV file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        return tmp.name

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def create_pdf_report(features, prediction, proba, audio_plots):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # [Previous PDF creation code remains the same...]
    return pdf.output(dest='S').encode('latin-1')

def plot_audio_features(y, sr):
    plots = []
    temp_dir = tempfile.gettempdir()
    
    # [Previous plotting code remains the same...]
    return plots

def extract_features(audio_path):
    features = {
        "mean_f0": 0, "std_f0": 0, "min_f0": 0, "max_f0": 0,
        "jitter": 0, "shimmer": 0, "hnr": 0, "nhr": 0,
        "rpde": 0, "dfa": 0, "spread1": 0, "spread2": 0,
        "d2": 0, "ppe": 0, "mfcc1": 0, "mfcc2": 0,
        "mfcc3": 0, "mfcc4": 0, "spectral_centroid": 0,
        "spectral_bandwidth": 0, "spectral_rolloff": 0
    }

    try:
        # Load audio with soundfile first to handle format
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:  # Convert stereo to mono if needed
            y = np.mean(y, axis=1)
            
        if len(y) < sr * 0.5:
            st.warning("Audio too short for analysis (minimum 0.5 seconds)")
            return None, None

        # [Rest of your feature extraction code...]
        return features, plot_audio_features(y, sr)

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None, None

def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("Record or upload a short 'ahhh' recording (3-5 seconds) for analysis")

    with st.sidebar:
        st.header("Instructions")
        st.markdown("1. Record using microphone or upload WAV file\n2. Click Analyze\n3. View results and download report")

    # Initialize variables
    audio_bytes = None
    uploaded_file = None
    
    st.subheader("1. Provide Audio Sample")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Record using microphone**")
        audio_bytes = audio_recorder(
            text="Click to record (5 seconds)",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=5.0
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
    
    with col2:
        st.markdown("**Or upload audio file**")
        uploaded_file = st.file_uploader(
            "Choose WAV file", 
            type=["wav"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")

    # Only show analyze button if we have audio
# parkinsons_voice_app.py
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy
import joblib
import warnings
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
from datetime import datetime
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import wave
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load("parkinsons_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'parkinsons_model.pkl' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def save_audio_bytes(audio_bytes):
    """Save raw audio bytes to a WAV file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        return tmp.name

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def create_pdf_report(features, prediction, proba, audio_plots):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Parkinson's Voice Analysis Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnostic Results", ln=1)
    pdf.set_font("Arial", size=12)
    
    pred_text = "Possible Parkinson's" if prediction else "Healthy"
    confidence = f"{proba*100:.1f}% confidence"
    risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low"
    
    pdf.cell(200, 10, txt=f"Prediction: {pred_text}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {confidence}", ln=1)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_level}", ln=1)
    pdf.ln(10)
    
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
    
    return pdf.output(dest='S').encode('latin-1')

def plot_audio_features(y, sr):
    plots = []
    temp_dir = tempfile.gettempdir()
    
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_path = os.path.join(temp_dir, f"waveform_{int(time.time())}.png")
    plt.savefig(waveform_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots.append(waveform_path)
    
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join(temp_dir, f"spectrogram_{int(time.time())}.png")
    plt.savefig(spectrogram_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots.append(spectrogram_path)
    
    return plots

def extract_features(audio_path):
    features = {
        "mean_f0": 0, "std_f0": 0, "min_f0": 0, "max_f0": 0,
        "jitter": 0, "shimmer": 0, "hnr": 0, "nhr": 0,
        "rpde": 0, "dfa": 0, "spread1": 0, "spread2": 0,
        "d2": 0, "ppe": 0, "mfcc1": 0, "mfcc2": 0,
        "mfcc3": 0, "mfcc4": 0, "spectral_centroid": 0,
        "spectral_bandwidth": 0, "spectral_rolloff": 0
    }

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        if len(y) < sr * 0.5:
            st.warning("Audio too short for analysis (minimum 0.5 seconds)")
            return None, None

        # Fundamental frequency estimation
        f0 = librosa.yin(y, fmin=50, fmax=500)
        f0 = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0) > 10:
            features.update({
                "mean_f0": np.mean(f0),
                "std_f0": np.std(f0),
                "min_f0": np.min(f0),
                "max_f0": np.max(f0),
                "ppe": entropy(f0) if len(f0) > 1 else 0,
                "spread1": np.std(f0),
                "spread2": np.var(f0),
                "d2": np.percentile(f0, 99),
            })

        # Jitter and shimmer (approximations)
        if len(f0) > 2:
            diffs = np.diff(f0)
            features["jitter"] = np.mean(np.abs(diffs)) / np.mean(f0)
            amp = librosa.feature.rms(y=y)[0]
            amp_diffs = np.diff(amp)
            features["shimmer"] = np.mean(np.abs(amp_diffs)) / np.mean(amp)

        # Harmonic-to-noise ratio
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        hnr = 10 * np.log10(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-6))
        features["hnr"] = hnr
        features["nhr"] = 1 / (hnr + 1e-6) if hnr > 0 else 0

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
        features.update({
            "mfcc1": np.mean(mfccs[0]),
            "mfcc2": np.mean(mfccs[1]),
            "mfcc3": np.mean(mfccs[2]),
            "mfcc4": np.mean(mfccs[3]),
        })

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        features.update({
            "spectral_centroid": np.mean(spectral_centroid),
            "spectral_bandwidth": np.mean(spectral_bandwidth),
            "spectral_rolloff": np.mean(spectral_rolloff),
            "rpde": entropy(spectral_centroid[0]),
            "dfa": librosa.feature.rms(y=y).mean(),
        })

        # Generate plots
        plots = plot_audio_features(y, sr)

        return features, plots

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None, None

def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("Record or upload a short 'ahhh' recording (3-5 seconds) for analysis")

    with st.sidebar:
        st.header("Instructions")
        st.markdown("1. Record using microphone or upload WAV file\n2. Click Analyze\n3. View results and download report")

    # Initialize variables
    audio_bytes = None
    uploaded_file = None
    
    st.subheader("1. Provide Audio Sample")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Record using microphone**")
        audio_bytes = audio_recorder(
            text="Click to record (5 seconds)",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=5.0
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
    
    with col2:
        st.markdown("**Or upload audio file**")
        uploaded_file = st.file_uploader(
            "Choose WAV file", 
            type=["wav"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")

    # Only show analyze button if we have audio
    if audio_bytes is not None or uploaded_file is not None:
        if st.button("Analyze Voice", type="primary", use_container_width=True):
            with st.spinner("Analyzing voice patterns..."):
                try:
                    # Save audio to proper WAV file
                    if audio_bytes:
                        audio_path = save_audio_bytes(audio_bytes)
                    else:
                        audio_path = save_uploaded_file(uploaded_file)
                    
                    # Rest of your analysis code...
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                finally:
                    # Clean up temp file
                    if 'audio_path' in locals():
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
    else:
        st.info("Please record or upload an audio file to analyze")

if __name__ == "__main__":
    main()
