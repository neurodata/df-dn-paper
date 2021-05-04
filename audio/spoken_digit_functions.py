import os

import cv2
import librosa
import numpy as np
import torchaudio.transforms as trans
import torch


def load_spoken_digit(path_recordings):
    file = os.listdir(path_recordings)

    audio_data = []  # audio data
    x_spec = []  # STFT spectrogram
    x_spec_mini = []  # resized image, 28*28
    y_number = []  # label of number
    y_speaker = []  # label of speaker
    a = trans.Spectrogram(n_fft=128, normalized=True)
    for i in file:
        x, sr = librosa.load(path_recordings + i, sr=8000)
        # x_stft = librosa.stft(x, n_fft=128)  # Extract STFT
        # x_stft_db = librosa.amplitude_to_db(abs(x_stft))
        x_stft_db = a(torch.tensor(x)).numpy()
        # Convert an amplitude spectrogram to dB-scaled spectrogram
        x_stft_db_mini = cv2.resize(x_stft_db, (32, 32))  # Resize into 28 by 28
        y_n = i[0]  # number
        y_s = i[2]  # first letter of speaker's name

        audio_data.append(x)
        x_spec.append(x_stft_db)
        x_spec_mini.append(x_stft_db_mini)
        y_number.append(y_n)
        y_speaker.append(y_s)

    x_spec_mini = np.array(x_spec_mini)
    y_number = np.array(y_number).astype(int)
    y_speaker = np.array(y_speaker)

    return x_spec_mini, y_number
