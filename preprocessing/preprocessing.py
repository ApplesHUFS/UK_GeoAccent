"""
오디오 전처리 및 SpecAugment 구현
"""

import torch
import numpy as np

class SpecAugment:
    """SpecAugment 구현"""
    def __init__(self, freq_mask_param=30, time_mask_param=40):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    
    def __call__(self, spectrogram):
        """
        Args:
            spectrogram: (freq, time) 형태의 멜 스펙트로그램
        Returns:
            augmented_spectrogram
        """
        import torch
        import random

        augmented_spectrogram = spectrogram.clone()

        # TODO: SpecAugment 적용
        # 1. FrequencyMasking 적용
        num_freq_bins = augmented_spectrogram.size(0)
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, max(0, num_freq_bins - f))
        augmented_spectrogram[f0:f0 + f, :] = 0

        # 2. TimeMasking 적용
        num_time_bins = augmented_spectrogram.size(1)
        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, max(0, num_time_bins - t))
        augmented_spectrogram[:, t0:t0 + t] = 0
        # 3. 결과 반환
        return augmented_spectrogram


class AudioPreprocessor:
    """오디오 전처리"""
    def __init__(self, sample_rate=16000, use_augment=False):
        self.sample_rate = sample_rate
        self.use_augment = use_augment
        if use_augment:
            self.augment = SpecAugment()
    
    def load_audio(self, audio_path):

        import soundfile as sf
        import numpy as np
        import torch
        import librosa

        """
        오디오 파일 로드 및 리샘플링
        
        Args:
            audio_path: 오디오 파일 경로
        
        Returns:
            waveform: (sample_rate * duration,) 형태의 텐서
        """
        # TODO: 구현
        # 1. 오디오 로드
        data, sr = sf.read(audio_path, always_2d=False)

        # 2. sample_rate 확인 및 필요시 리샘플링
        target_sr = self.sample_rate if hasattr(self, "sample_rate") else 16000
        if sr != target_sr:
            # librosa는 (N,) 형태만 리샘플 가능
            if isinstance(data, np.ndarray) and data.ndim == 2:
                data = data.mean(axis=1)  # 스테레오 → 모노 변환
            data = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        # numpy → torch 변환
        waveform = torch.tensor(data, dtype=torch.float32)

        # 3. 모노 채널로 변환 (스테레오면)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # 4. 반환
        return waveform, sr
    
    def normalize_audio(self, waveform):
        """
        오디오 정규화 (평균 0, 표준편차 1)
        
        Args:
            waveform: 원본 waveform
        
        Returns:
            normalized_waveform
        """
        # TODO: 구현
        import torch
        # mean과 std를 계산하여 정규화
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        
        if std < 1e-8: # 표준편차가 0에 가까울 때 (무음 등)
            return waveform - mean
        else: 
            return (waveform - mean) / (std + 1e-8)