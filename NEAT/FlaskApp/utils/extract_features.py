# Feature extraction logic goes here
import librosa
import numpy as np
from scipy.stats import variation


def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # F1 (fundamental frequency)
    F1, _, _ = librosa.pyin(y, fmin=50, fmax=400)
    avg_F1 = np.nanmean(F1)

    # Jitter approximation: relative f0 variation
    F1_diff = np.diff(F1)
    jitter_s = np.nanmean(np.abs(F1_diff / F1[1:])) if F1 is not None else 0

    # Shimmer approximation: energy envelope variability
    energy = librosa.feature.rms(y=y)[0]
    shimmer = np.mean(np.abs(np.diff(energy))) if len(energy) > 1 else 0

    # HNR approximation using harmonic-to-noise ratio from autocorrelation
    autocorr = librosa.autocorrelate(y)
    mean_hnr = 10 * np.log10(np.max(autocorr) / (np.mean(autocorr) + 1e-6))

    # MFCCs (mean over time frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    # Return both acoustic and MFCCs
    features = [avg_F1, jitter_s, shimmer, mean_hnr]
    return features, mfcc_means