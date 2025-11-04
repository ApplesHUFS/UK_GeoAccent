# ============================================================================
# 👤 PERSON A: 파일 2: data/preprocessing.py
# ============================================================================

"""
오디오 전처리 및 SpecAugment 구현
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

class SpecAugment:
    """SpecAugment 구현"""
    def __init__(self, freq_mask_param=30, time_mask_param=40):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
    
    def __call__(self, spectrogram):
        """
        Args:
            spectrogram: (freq, time) 형태의 멜 스펙트로그램
        Returns:
            augmented_spectrogram
        """
        # TODO: SpecAugment 적용
        # 1. FrequencyMasking 적용
        # 2. TimeMasking 적용
        # 3. 결과 반환
        pass

class AudioPreprocessor:
    """오디오 전처리"""
    def __init__(self, sample_rate=16000, use_augment=False):
        self.sample_rate = sample_rate
        self.use_augment = use_augment
        if use_augment:
            self.augment = SpecAugment()
    
    def load_audio(self, audio_path):
        """
        오디오 파일 로드 및 리샘플링
        
        Args:
            audio_path: 오디오 파일 경로
        
        Returns:
            waveform: (sample_rate * duration,) 형태의 텐서
        """
        # TODO: 구현
        # 1. torchaudio.load()로 오디오 로드
        # 2. sample_rate 확인 및 필요시 리샘플링
        # 3. 모노 채널로 변환 (스테레오면)
        # 4. 반환
        pass
    
    def normalize_audio(self, waveform):
        """
        오디오 정규화 (평균 0, 표준편차 1)
        
        Args:
            waveform: 원본 waveform
        
        Returns:
            normalized_waveform
        """
        # TODO: 구현
        # mean과 std를 계산하여 정규화
        pass