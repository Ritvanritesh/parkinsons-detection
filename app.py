import streamlit as st
import parselmouth
import numpy as np
import pickle
import os

# Load trained model
model_path = "parkinsons_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please upload 'parkinsons_model.pkl'.")
else:
    model = pickle.load(open(model_path, "rb"))

    def extract_features(file_path):
        try:
            # Ensure file is valid WAV and mono
            snd = parselmouth.Sound(file_path)

            if snd.n_channels != 1:
                st.warning("Converting stereo to mono...")
                samples = np.mean(snd.values, axis=0)
                snd = parselmouth.Sound(values=samples, sampling_frequency=snd.sampling_frequency)

            pitch = snd.to_pitch()
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_value = parselmouth.praat.call(hnr, "Get mean", 0, 0)

            return np.array([jitter, shimmer, hnr_value])
        except Exception as e:
            st.error(f"⚠️ Feature extraction failed: {e}")
            return None

    st.title("🧠 Parkinson's Detection from Voice")
    st.write("Upload a `.wav` file of a person saying 'aaaah' for 3–5 seconds.")

    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        features = extract_features("temp_audio.wav")

        if features is not None:
            prediction = model.predict([features])

            if prediction[0] == 1:
                st.error("🧠 Parkinson's Detected!")
            else:
                st.success("✅ No Parkinson's Detected!")
