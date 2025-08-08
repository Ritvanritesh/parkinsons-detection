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

# --- Model Loading ---
@st.cache_resource
def load_model():
    if not os.path.exists("parkinsons_model.pkl"):
        st.error("❌ Model file not found. Please add 'parkinsons_model.pkl' to your app directory.")
        st.stop()
    return joblib.load("parkinsons_model.pkl")

# --- Audio Processing ---
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"File save failed: {str(e)}")
        return None

def extract_features(audio_path):
    features = {
        "MDVP:Fo(Hz)": 0, "MDVP:Fhi(Hz)": 0, "MDVP:Flo(Hz)": 0,
        # ... (keep your existing feature dictionary)
    }
    
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        snd = parselmouth.Sound(audio_path)
        
        # ... (keep your existing feature extraction logic)
        
        return features, plot_audio_features(y, sr)
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None, None
    finally:
        try:
            os.unlink(audio_path)
        except:
            pass

# --- Main App ---
def main():
    st.title("🧠 Parkinson's Voice Analysis")
    
    # Audio Input Section
    audio_source = None
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Recording")
        audio_bytes = audio_recorder(text="", pause_threshold=5.0)
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_source = tmp.name
            st.audio(audio_bytes, format="audio/wav")
    
    with col2:
        st.subheader("Or Upload File")
        uploaded_file = st.file_uploader("Choose WAV", type=["wav"], label_visibility="collapsed")
        if uploaded_file:
            audio_source = save_uploaded_file(uploaded_file)
            st.audio(uploaded_file, format="audio/wav")
    
    # Analysis Section
    if st.button("Analyze Voice") and audio_source:
        with st.spinner("Analyzing..."):
            features, plots = extract_features(audio_source)
            
            if features:
                model = load_model()
                df = pd.DataFrame([features])
                
                prediction = model.predict(df)[0]
                proba = model.predict_proba(df)[0][1]
                
                # Display Results
                st.success("Analysis Complete!")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", 
                           "🧠 Possible Parkinson's" if prediction else "✅ Healthy",
                           f"{proba*100:.1f}%")
                col2.metric("Confidence Level", 
                           "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low")
                
                # Visualization
                with st.expander("Audio Analysis"):
                    st.image(plots[0], caption="Waveform")
                    st.image(plots[1], caption="Spectrogram")
                
                # Download Report
                pdf = create_pdf_report(features, prediction, proba, plots)
                st.download_button(
                    label="📄 Download Full Report",
                    data=pdf,
                    file_name="parkinson_analysis.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
