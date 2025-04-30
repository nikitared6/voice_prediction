import librosa
import numpy as np
import parselmouth
import opensmile
import pandas as pd
from scipy.stats import entropy

def extract_features(audio_path, sr=22050):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract fundamental frequency (F0) using Praat
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]  # Remove unvoiced regions

    if len(f0_values) > 0:
        fo_mean = np.mean(f0_values)
        fo_max = np.max(f0_values)
        fo_min = np.min(f0_values)
    else:
        fo_mean = fo_max = fo_min = 0

    # Jitter and Shimmer features using Praat
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    
    jitter_local = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0.0001, 0.02, 1.3)
    jitter_absolute = parselmouth.praat.call([snd, point_process], "Get jitter (absolute)", 0.0001, 0.02, 1.3)
    jitter_rap = parselmouth.praat.call([snd, point_process], "Get jitter (rap)", 0.0001, 0.02, 1.3)
    jitter_ppq = parselmouth.praat.call([snd, point_process], "Get jitter (ppq5)", 0.0001, 0.02, 1.3)
    jitter_ddp = 3 * jitter_rap  # Approximate formula

    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0.0001, 0.02, 1.3)
    shimmer_local_db = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0.0001, 0.02, 1.3)
    shimmer_apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0.0001, 0.02, 1.3)
    shimmer_apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0.0001, 0.02, 1.3)
    shimmer_apq = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0.0001, 0.02, 1.3)
    shimmer_dda = 3 * shimmer_apq3  # Approximate formula

    # Noise-to-Harmonic Ratio (NHR) and Harmonics-to-Noise Ratio (HNR)
    hnr = parselmouth.praat.call(snd, "Get harmonics-to-noise ratio", 0.0, 75.0, 300.0)
    nhr = 1 / hnr if hnr != 0 else 0

    # Recurrence Period Density Entropy (RPDE), DFA, Spread, and PPE require OpenSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    
    smile_features = smile.process_file(audio_path)

    rpde = smile_features.get("F0final_sma_de_pctl_1", [0])[0]
    dfa = smile_features.get("ShimmerLocal_sma_de_stddev", [0])[0]
    spread1 = smile_features.get("F0final_sma_amean", [0])[0] - np.std(f0_values) if len(f0_values) > 0 else 0
    spread2 = smile_features.get("F0final_sma_de_range", [0])[0]
    d2 = smile_features.get("pcm_loudness_sma_de_amean", [0])[0]

    # Pitch Period Entropy (PPE)
    ppe = entropy(f0_values) if len(f0_values) > 1 else 0

    # Combine all extracted features into a dictionary
    features = {
        "MDVP:Fo(Hz)": fo_mean,
        "MDVP:Fhi(Hz)": fo_max,
        "MDVP:Flo(Hz)": fo_min,
        "MDVP:Jitter(%)": jitter_local,
        "MDVP:Jitter(Abs)": jitter_absolute,
        "MDVP:RAP": jitter_rap,
        "MDVP:PPQ": jitter_ppq,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_local_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": shimmer_apq,
        "Shimmer:DDA": shimmer_dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe,
    }

    return features

# Example usage
audio_path = "AudioFiles/recorded_audio.wav"  # Replace with actual file path
features = extract_features(audio_path)
df = pd.DataFrame([features])
print(df)