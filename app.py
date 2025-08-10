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
