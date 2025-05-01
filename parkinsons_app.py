import streamlit as st
import librosa
import numpy as np
import pickle
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.stats import entropy

# Constants
UPLOAD_PATH = "AudioFiles/uploaded_audio.wav"
UPLOAD_DIR = "AudioFiles"

# Ensure the directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load model and scaler
model = joblib.load('models/xgb_clf_new.joblib')
scaler = joblib.load('models/scaler.joblib')

# Save uploaded file
def save_uploaded_file(uploadedfile):
    try:
        file_path = os.path.join(UPLOAD_DIR, uploadedfile.name)
        with open(file_path, "wb") as f:
            f.write(uploadedfile.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

# Feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = {}
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]

        features['MDVP:Fo(Hz)'] = np.mean(pitches) if len(pitches) > 0 else np.nan
        features['MDVP:Fhi(Hz)'] = np.max(pitches) if len(pitches) > 0 else np.nan
        features['MDVP:Flo(Hz)'] = np.min(pitches) if len(pitches) > 0 else np.nan

        zero_crossings = librosa.zero_crossings(y, pad=False)
        jitter_std = np.std(zero_crossings)
        jitter_mean = np.mean(zero_crossings)
        features['MDVP:Jitter(%)'] = jitter_std / jitter_mean if jitter_mean != 0 else np.nan
        features['MDVP:Jitter(Abs)'] = jitter_std
        features['MDVP:RAP'] = jitter_std / (len(zero_crossings) + 1e-6)
        features['MDVP:PPQ'] = jitter_std / np.sqrt(len(zero_crossings) + 1e-6)
        features['Jitter:DDP'] = jitter_std * 3

        amplitude = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        shimmer_std = np.std(amplitude)
        shimmer_mean = np.mean(amplitude)
        features['MDVP:Shimmer'] = shimmer_std / shimmer_mean if shimmer_mean != 0 else np.nan
        features['MDVP:Shimmer(dB)'] = shimmer_std
        features['Shimmer:APQ3'] = shimmer_std / 3
        features['Shimmer:APQ5'] = shimmer_std / 5
        features['MDVP:APQ'] = shimmer_std / len(amplitude)
        features['Shimmer:DDA'] = shimmer_std * 3

        harmonic, percussive = librosa.effects.hpss(y)
        features['NHR'] = np.mean(percussive) / (np.mean(harmonic) + 1e-6)
        features['HNR'] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        features['RPDE'] = entropy(pitches) if len(pitches) > 0 else np.nan
        features['DFA'] = librosa.feature.rms(y=y).mean()

        features['spread1'] = np.std(pitches) if len(pitches) > 0 else np.nan
        features['spread2'] = np.var(pitches) if len(pitches) > 0 else np.nan
        features['D2'] = np.percentile(pitches, 99) if len(pitches) > 0 else np.nan
        features['PPE'] = np.mean(np.abs(pitches - np.mean(pitches))) if len(pitches) > 0 else np.nan

        return features
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# Streamlit app
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
st.title("🧠 Parkinson's Disease Detection from Voice")
st.markdown("Analyze voice recordings to detect the likelihood of Parkinson's disease using machine learning.")

# Upload file
uploaded_file = st.file_uploader("Upload a voice recording (.wav or .mp3)", type=["wav", "mp3"])
if uploaded_file:
    audio_path = save_uploaded_file(uploaded_file)
    st.audio(audio_path, format="audio/wav")

    st.markdown("---")
    st.subheader("📊 Prediction Results and Extracted Features")

    if model and scaler:
        features = extract_features(audio_path)
        if features is not None:
            try:
                feature_array = np.array(list(features.values())).reshape(1, -1)
                scaled_features = scaler.transform(feature_array)
                prediction = model.predict(scaled_features)[0]

                if prediction == 1:
                    st.error("🔴 High likelihood of Parkinson's detected. Please consult a doctor.")
                else:
                    st.success("🟢 Low likelihood of Parkinson's detected. Stay healthy!")

                st.markdown("### Extracted Feature Values")
                feature_df = pd.DataFrame([features], index=["Values"]).T
                feature_df.columns = ["Feature Value"]
                st.dataframe(feature_df.style.format("{:.4f}"))
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.warning("Feature extraction failed.")
    else:
        st.warning("Model or scaler file is missing!")

# Footer
st.markdown("---")
st.write("🔍 **Note:** This app is for demonstration purposes and is not a substitute for professional medical advice.")
