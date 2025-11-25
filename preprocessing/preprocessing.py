"""
preprocessing/preprocessing.py
Audio preprocessing and SpecAugment
"""

import torch
import numpy as np

class SpecAugment:
    """SpecAugment implementation"""
    def __init__(self, freq_mask_param=30, time_mask_param=40):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spectrogram):
        """
        Args:
            spectrogram: Mel spectrogram of shape (freq, time)
        Returns:
            augmented_spectrogram
        """
        import random

        augmented_spectrogram = spectrogram.clone()

        # Frequency masking
        num_freq_bins = augmented_spectrogram.size(0)
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, max(0, num_freq_bins - f))
        augmented_spectrogram[f0:f0 + f, :] = 0

        # Time masking
        num_time_bins = augmented_spectrogram.size(1)
        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, max(0, num_time_bins - t))
        augmented_spectrogram[:, t0:t0 + t] = 0

        return augmented_spectrogram


class AudioPreprocessor:
    """Audio preprocessing"""
    def __init__(self, sample_rate=16000, use_augment=False):
        self.sample_rate = sample_rate
        self.use_augment = use_augment
        if use_augment:
            self.augment = SpecAugment()
    
    def load_audio(self, audio_path):
        """
        Load and resample audio file
        """
        import soundfile as sf
        import librosa

        # Load audio
        data, sr = sf.read(audio_path, always_2d=False)

        # Resample if needed
        target_sr = self.sample_rate
        if sr != target_sr:
            if isinstance(data, np.ndarray) and data.ndim == 2:
                data = data.mean(axis=1)  # Stereo to mono
            data = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        waveform = torch.tensor(data, dtype=torch.float32)

        # Ensure mono channel
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return waveform, sr
    
    def normalize_audio(self, waveform):
        """
        Normalize audio to zero mean and unit variance
        """
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        
        if std < 1e-8:  # near silence
            return waveform - mean
        else: 
            return (waveform - mean) / (std + 1e-8)
