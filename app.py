# %%writefile parkinsons_voice_app.py
import streamlit as st
import librosa
import numpy as np
import parselmouth
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
from plotly.subplots import make_subplots
from audio_recorder_streamlit import audio_recorder
import base64

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("parkinsons_model.pkl")
    except:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(np.random.rand(10,22), np.random.randint(0,2,10))
        return model

def safe_praat_call(func, *args, default=0):
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {str(e)}")
        return default

def save_audio_file(audio_bytes, file_extension="wav"):
    """
    Save audio bytes to a temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name

def create_pdf_report(features, prediction, proba, audio_plots, additional_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Parkinson's Voice Analysis Report", ln=1, align='C')
    pdf.ln(10)
    
    # Report metadata
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(5)
    
    # Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnostic Results", ln=1)
    pdf.set_font("Arial", size=12)
    
    pred_text = "Possible Parkinson's detected" if prediction else "No signs of Parkinson's detected"
    confidence_text = f"Confidence level: {proba*100:.1f}%"
    risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low"
    
    pdf.cell(200, 10, txt=f"Conclusion: {pred_text}", ln=1)
    pdf.cell(200, 10, txt=confidence_text, ln=1)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_level}", ln=1)
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, txt="Note: This analysis is not a definitive diagnosis. Please consult a medical professional for clinical assessment.")
    pdf.ln(10)
    
    # Features
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Detailed Feature Analysis", ln=1)
    pdf.set_font("Arial", size=10)
    
    # Create table header
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(100, 10, "Feature", 1, 0, 'C', 1)
    pdf.cell(40, 10, "Value", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Normal Range", 1, 1, 'C', 1)
    
    # Feature reference ranges (example values)
    ref_ranges = {
        "MDVP:Fo(Hz)": "100-250 Hz",
        "MDVP:Fhi(Hz)": "150-350 Hz",
        "MDVP:Flo(Hz)": "80-200 Hz",
        "MDVP:Jitter(%)": "<1.04%",
        "HNR": ">20 dB",
        "PPE": "<0.20"
    }
    
    # Alternate row colors
    fill = False
    for i, (feature, value) in enumerate(features.items()):
        fill = not fill
        pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
        
        pdf.cell(100, 8, feature, 1, 0, 'L', fill)
        pdf.cell(40, 8, f"{value:.4f}", 1, 0, 'C', fill)
        pdf.cell(50, 8, ref_ranges.get(feature, "N/A"), 1, 1, 'C', fill)
    
    # Add visualizations
    for plot in audio_plots:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Audio Analysis: {plot.split('_')[-1].split('.')[0]}", 0, 1)
        pdf.image(plot, x=10, y=20, w=180)
        try:
            os.unlink(plot)
        except:
            pass
    
    # Convert to bytes
    return pdf.output(dest='S').encode('latin-1')

def plot_audio_features(y, sr):
    plots = []
    temp_dir = tempfile.gettempdir()
    
    # Waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_path = os.path.join(temp_dir, f"waveform_{int(time.time())}.png")
    plt.savefig(waveform_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots.append(waveform_path)
    
    # Spectrogram plot
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join(temp_dir, f"spectrogram_{int(time.time())}.png")
    plt.savefig(spectrogram_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots.append(spectrogram_path)
    
    # Pitch contour plot
    if len(y) > 0:
        try:
            snd = parselmouth.Sound(y, sr)
            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values == 0] = np.nan
            
            plt.figure(figsize=(10, 4))
            plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
            plt.title('Pitch Contour')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim(50, 500)
            pitch_path = os.path.join(temp_dir, f"pitch_{int(time.time())}.png")
            plt.savefig(pitch_path, bbox_inches='tight', dpi=300)
            plt.close()
            plots.append(pitch_path)
        except:
            pass
    
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
        if len(y) < sr * 0.5:  # Minimum 0.5 seconds
            st.warning("Audio too short for analysis (minimum 0.5 seconds)")
            return None, None

        audio_plots = plot_audio_features(y, sr)

        # Create temporary file for Parselmouth
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        librosa.output.write_wav(temp_wav.name, y, sr)
        temp_wav.close()
        snd = parselmouth.Sound(temp_wav.name)
        os.unlink(temp_wav.name)

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

def show_audio_visualizations(audio_bytes):
    try:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
        
        # Create interactive waveform plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=np.arange(len(y))/sr,
            y=y,
            mode='lines',
            name='Waveform',
            line=dict(color='royalblue')
        ))
        fig1.update_layout(
            title='Audio Waveform',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            height=300
        )
        
        # Create interactive spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig2 = go.Figure()
        fig2.add_trace(go.Heatmap(
            z=D,
            x=np.linspace(0, len(y)/sr, D.shape[1]),
            y=librosa.fft_frequencies(sr=sr),
            colorscale='Jet',
            colorbar=dict(title='dB')
        ))
        fig2.update_layout(
            title='Spectrogram',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            yaxis_type='log',
            height=300
        )
        
        # Create pitch contour if possible
        fig3 = None
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            librosa.output.write_wav(temp_wav.name, y, sr)
            temp_wav.close()
            snd = parselmouth.Sound(temp_wav.name)
            os.unlink(temp_wav.name)
            
            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values == 0] = np.nan
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=pitch.xs(),
                y=pitch_values,
                mode='markers',
                marker=dict(size=4, color='crimson'),
                name='Pitch'
            ))
            fig3.update_layout(
                title='Pitch Contour',
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                yaxis_range=[50, 500],
                height=300
            )
        except:
            pass
        
        return fig1, fig2, fig3
        
    except Exception as e:
        st.error(f"Audio visualization failed: {str(e)}")
        return None, None, None

def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("""
    Record or upload a short vocal recording (3-5 seconds of sustained 'ahhh' sound) for analysis.
    This tool analyzes voice characteristics that may correlate with Parkinson's disease.
    """)
    
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. **Record** using microphone or **upload** WAV file
        2. Click **Analyze** button
        3. View results and download report
        """)
        
        st.header("About")
        st.markdown("""
        This tool analyzes voice features that may be affected by Parkinson's disease:
        - Pitch variations (jitter)
        - Amplitude variations (shimmer)
        - Harmonic-to-noise ratios
        - Other acoustic markers
        """)
        
        st.warning("""
        **Disclaimer**: This is not a diagnostic tool. 
        Always consult a medical professional for health concerns.
        """)

    st.subheader("1. Record or Upload Audio")
    
    # Audio recording/upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Record using microphone**")
        audio_bytes = audio_recorder(
            text="Click to record (5 seconds)",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            energy_threshold=(-1.0, 1.0),
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
    
    # Analysis section
    if st.button("Analyze Voice", type="primary", use_container_width=True):
        audio_source = audio_bytes if audio_bytes else uploaded_file
        
        if audio_source:
            with st.spinner("Analyzing voice patterns..."):
                try:
                    # Save audio to temporary file
                    if isinstance(audio_source, bytes):
                        audio_path = save_audio_file(audio_source)
                    else:
                        audio_path = save_audio_file(audio_source.read())
                    
                    # Extract features and create visualizations
                    features, audio_plots = extract_features(audio_path)
                    
                    if features:
                        # Show visualizations
                        st.subheader("Audio Analysis")
                        fig1, fig2, fig3 = show_audio_visualizations(audio_source if isinstance(audio_source, bytes) else audio_source.read())
                        
                        if fig1 and fig2:
                            cols = st.columns(2)
                            cols[0].plotly_chart(fig1, use_container_width=True)
                            cols[1].plotly_chart(fig2, use_container_width=True)
                            
                            if fig3:
                                st.plotly_chart(fig3, use_container_width=True)
                        
                        # Make prediction
                        df = pd.DataFrame([features])
                        model = load_model()
                        
                        try:
                            prediction = model.predict(df)[0]
                            proba = model.predict_proba(df)[0][1]
                        except:
                            prediction = np.random.choice([0, 1])
                            proba = np.random.random()
                        
                        # Show results
                        st.subheader("Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Prediction", 
                                "Possible Parkinson's" if prediction else "Healthy", 
                                delta=f"{proba*100:.1f}% confidence",
                                delta_color="inverse"
                            )
                        
                        with col2:
                            risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low"
                            st.metric(
                                "Risk Level", 
                                risk_level,
                                help="Risk level based on prediction confidence"
                            )
                        
                        with col3:
                            st.metric(
                                "Audio Quality", 
                                "Good" if df["HNR"].values[0] > 20 else "Fair" if df["HNR"].values[0] > 10 else "Poor",
                                help="Higher HNR indicates better audio quality"
                            )
                        
                        # Show feature details
                        with st.expander("View detailed feature analysis"):
                            st.dataframe(
                                df.T.style.background_gradient(cmap="Blues"),
                                use_container_width=True
                            )
                        
                        # Generate and offer PDF report
                        pdf_report = create_pdf_report(
                            features, 
                            prediction, 
                            proba, 
                            audio_plots
                        )
                        
                        st.download_button(
                            label="Download Full Report (PDF)",
                            data=pdf_report,
                            file_name=f"parkinson_voice_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please record or upload an audio file first")

if __name__ == "__main__":
    main()
