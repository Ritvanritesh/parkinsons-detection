import streamlit as st
import numpy as np
import pandas as pd
import pickle
import librosa
import soundfile as sf
import tempfile
import io
import sounddevice as sd
from scipy.stats import skew, kurtosis

# --------------------------
# Load trained model
# --------------------------
with open("parkinsons_model.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------
# Feature extraction function
# --------------------------
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Basic audio features
        features = {}
        features['mean_pitch'] = np.mean(librosa.yin(y, fmin=50, fmax=500))
        features['std_pitch'] = np.std(librosa.yin(y, fmin=50, fmax=500))

        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        # Jitter & Shimmer approximations
        zero_crossings = librosa.zero_crossings(y, pad=False)
        features['zero_crossing_rate'] = np.mean(zero_crossings)

        # Harmonic-to-noise ratio (approximation)
        S, phase = librosa.magphase(librosa.stft(y))
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        features['hnr'] = np.mean(harmonic**2) / (np.mean(noise**2) + 1e-6)

        # Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # Skewness & Kurtosis
        features['skewness'] = skew(y)
        features['kurtosis'] = kurtosis(y)

        return np.array(list(features.values())).reshape(1, -1)

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# --------------------------
# Streamlit UI
# --------------------------
st.title("🧠 Parkinson's Voice Analysis")
st.write("Upload a voice recording or record your voice to detect Parkinson's Disease using ML.")

# File upload
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

# Live recording
if st.button("🎙 Record 5 seconds"):
    st.info("Recording... Speak now!")
    duration = 5
    fs = 22050
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete!")

    # Save temp audio file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, recording, fs)
    uploaded_file = temp_file

# Prediction
if uploaded_file is not None:
    if isinstance(uploaded_file, tempfile._TemporaryFileWrapper):
        audio_path = uploaded_file.name
    else:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path.write(uploaded_file.read())
        audio_path = temp_path.name

    features = extract_features(audio_path)
    if features is not None:
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.error("⚠️ Parkinson's Detected")
        else:
            st.success("✅ No Parkinson's Detected")
