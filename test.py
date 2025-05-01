import librosa
import numpy as np
import parselmouth
import pandas as pd
from scipy.stats import entropy

def extract_features(audio_path, sr=22050):
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Use Parselmouth (Praat) for advanced voice features
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]

    # Fundamental frequency stats
    fo_mean = np.mean(f0_values) if len(f0_values) > 0 else 0
    fo_max = np.max(f0_values) if len(f0_values) > 0 else 0
    fo_min = np.min(f0_values) if len(f0_values) > 0 else 0

    # Jitter and Shimmer via Praat
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    jitter_local = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0.0001, 0.02, 1.3)
    jitter_absolute = parselmouth.praat.call([snd, point_process], "Get jitter (absolute)", 0.0001, 0.02, 1.3)
    jitter_rap = parselmouth.praat.call([snd, point_process], "Get jitter (rap)", 0.0001, 0.02, 1.3)
    jitter_ppq = parselmouth.praat.call([snd, point_process], "Get jitter (ppq5)", 0.0001, 0.02, 1.3)
    jitter_ddp = 3 * jitter_rap

    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0.0001, 0.02, 1.3)
    shimmer_local_db = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0.0001, 0.02, 1.3)
    shimmer_apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0.0001, 0.02, 1.3)
    shimmer_apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0.0001, 0.02, 1.3)
    shimmer_apq = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0.0001, 0.02, 1.3)
    shimmer_dda = 3 * shimmer_apq3

    # Harmonics-to-noise ratio and inverse for NHR
    hnr = parselmouth.praat.call(snd, "Get harmonics-to-noise ratio", 0.0, 75.0, 300.0)
    nhr = 1 / hnr if hnr != 0 else 0

    # Additional pitch-based metrics
    spread1 = np.std(f0_values) if len(f0_values) > 0 else 0
    spread2 = np.var(f0_values) if len(f0_values) > 0 else 0
    d2 = np.percentile(f0_values, 99) if len(f0_values) > 0 else 0
    ppe = entropy(f0_values) if len(f0_values) > 1 else 0
    dfa = librosa.feature.rms(y=y).mean()  # Simplified DFA approximation

    # Assemble features into a dictionary
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
        "RPDE": spread1,  # Approximation in absence of OpenSMILE
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe,
    }

    return features
