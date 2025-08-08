import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import soundfile as sf
from streamlit_audio_recorder import audio_recorder

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    return joblib.load("parkinsons_model.pkl")

model = load_model()

# --------------------------
# Extract 22 Vocal Features
# --------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    features = {}
    features['mean_freq'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['sd_freq'] = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['max_freq'] = np.max(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['min_freq'] = np.min(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['mean_amp'] = np.mean(librosa.feature.rms(y=y))
    features['sd_amp'] = np.std(librosa.feature.rms(y=y))
    features['max_amp'] = np.max(librosa.feature.rms(y=y))
    features['min_amp'] = np.min(librosa.feature.rms(y=y))
    features['mean_pitch'] = np.mean(librosa.yin(y, fmin=50, fmax=500))
    features['sd_pitch'] = np.std(librosa.yin(y, fmin=50, fmax=500))
    features['max_pitch'] = np.max(librosa.yin(y, fmin=50, fmax=500))
    features['min_pitch'] = np.min(librosa.yin(y, fmin=50, fmax=500))
    features['mfcc_mean'] = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    features['mfcc_sd'] = np.std(librosa.feature.mfcc(y=y, sr=sr))
    features['mfcc_max'] = np.max(librosa.feature.mfcc(y=y, sr=sr))
    features['mfcc_min'] = np.min(librosa.feature.mfcc(y=y, sr=sr))
    features['zero_crossing'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['rolloff_sd'] = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['chroma_mean'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features['chroma_sd'] = np.std(librosa.feature.chroma_stft(y=y, sr=sr))
    features['tonnetz_mean'] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))

    return pd.DataFrame([features])

# --------------------------
# Streamlit UI
# --------------------------
st.title("🧠 Parkinson's Voice Analysis")
st.write("Upload or record your voice to check for potential signs of Parkinson's Disease (Early Detection).")

# Audio Upload
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

# Audio Recording
st.write("Or record your voice below:")
recorded_audio = audio_recorder()

if recorded_audio:
    with open("recorded.wav", "wb") as f:
        f.write(recorded_audio)
    uploaded_file = "recorded.wav"

# Prediction
if uploaded_file:
    if isinstance(uploaded_file, str):  # recorded audio
        file_path = uploaded_file
    else:  # uploaded file
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    with st.spinner("Extracting features and making prediction..."):
        features_df = extract_features(file_path)
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1] * 100

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The model detects possible Parkinson's signs with {probability:.2f}% confidence.")
    else:
        st.success(f"✅ The model detects no significant Parkinson's signs with {probability:.2f}% confidence.")

    st.write("### Extracted Features")
    st.dataframe(features_df)

