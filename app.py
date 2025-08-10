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
    
    # [Previous UI code remains the same until the analysis button...]
    
    if st.button("Analyze Voice", type="primary", use_container_width=True):
        if audio_bytes or uploaded_file:
            with st.spinner("Analyzing voice patterns..."):
                try:
                    # Handle audio bytes properly
                    if audio_bytes:
                        if isinstance(audio_bytes, bytes):
                            audio_path = save_audio_bytes(audio_bytes)
                        else:
                            # Convert other audio types to bytes if needed
                            audio_path = save_audio_bytes(bytes(audio_bytes))
                    else:
                        audio_path = save_uploaded_file(uploaded_file)
                    
                    # [Rest of your analysis code...]
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                finally:
                    # Clean up temp file
                    if 'audio_path' in locals():
                        try:
                            os.unlink(audio_path)
                        except:
                            pass

if __name__ == "__main__":
    main()
