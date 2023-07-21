import os
import csv
import json
import numpy as np
import librosa

def extract_audio_features(signal, sr):
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=sr)
    rmse = librosa.feature.rms(y=signal)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)

    tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)

    features = [tempo] + list(np.mean(chroma_stft, axis=1)) + list(np.mean(rmse, axis=1)) + list(
        np.mean(spectral_centroid, axis=1)) + list(np.mean(spectral_bandwidth, axis=1)) + list(
        np.mean(rolloff, axis=1)) + list(np.mean(zero_crossing_rate, axis=1)) + list(
        np.mean(mfcc, axis=1)) + list(np.mean(mfcc_delta, axis=1)) + list(
        np.mean(tonnetz, axis=1)) + list(np.mean(spectral_contrast, axis=1))

    return features

# Se incarca fisierul CSV
with open('features.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Pentru fiecare fisier in parte se ruleaza functia 'extract_audio_features'
    for filename in os.listdir('data/test_wav'):
        if filename.endswith('.wav'):
            filepath = os.path.join('data/test_wav', filename)

            # Se incarca file audio si se extrag caracteristicile
            signal, sr = librosa.load(filepath)
            extracted_features = extract_audio_features(signal, sr)

            # Se scriu caracteristicile in fisierul CSV
            writer.writerow(extracted_features)
