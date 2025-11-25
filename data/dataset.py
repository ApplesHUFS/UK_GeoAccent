"""
data/dataset.py
Custom PyTorch Dataset 구현 (수정됨)
"""

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from datasets import load_from_disk

# utils/config.py에서 설정 가져오기
from utils.config import (
    REGION_LABELS, 
    GENDER_LABELS,
    REGION_COORDS, 
    normalize_coords
)
from preprocessing.preprocessing import AudioPreprocessor

class EnglishDialectsDataset(Dataset):
    """
    English Dialects 데이터셋
    
    HuggingFace Dataset 포맷을 처리하며, Wav2Vec2 모델 입력을 위해
    Raw Waveform을 반환합니다.
    """
    
    def __init__(self, split='train', use_augment=False, 
                 data_dir="./data/english_dialects", audio_sample_rate=16000):
        super().__init__()
        self.split = split
        self.use_augment = use_augment
        self.audio_sample_rate = audio_sample_rate
        
        # 1. 데이터셋 로드
        local_path = os.path.join(data_dir, split)
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Dataset not found at {local_path}. "
                f"Please run download script or check data_dir path."
            )
        
        try:
            self.dataset = load_from_disk(local_path)
            print(f"✅ Loaded {split} split: {len(self.dataset)} samples")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {local_path}: {e}")
        
        # 2. 전처리기 초기화
        # Preprocessor 클래스는 정규화(normalize_audio) 용도로만 사용합니다.
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.audio_sample_rate,
            use_augment=use_augment
        )
    
    def _parse_label(self, config_name):
        """
        레이블 파싱: 'irish_male' -> (region_id, gender_id, coords)
        """
        try:
            parts = config_name.split('_')
            region = parts[0]  # 'irish'
            gender = parts[1]  # 'male'
            
            region_id = REGION_LABELS.get(region, 0)
            gender_id = GENDER_LABELS.get(gender, 0)
            
            # 지역 좌표 가져오기 및 정규화
            lat, lon = REGION_COORDS.get(region, (0.0, 0.0))
            n_lat, n_lon = normalize_coords(lat, lon)
            coords = (n_lat, n_lon)
            
            return region_id, gender_id, coords
            
        except Exception as e:
            print(f"⚠️ Label parsing error for {config_name}: {e}")
            return 0, 0, (0.0, 0.0)

    def _process_waveform(self, audio_data, sr):
        """
        리샘플링 및 텐서 변환
        """
        # 1. 리샘플링 (필요한 경우)
        if sr != self.audio_sample_rate:
            # librosa.resample은 numpy array를 입력으로 받음
            audio_data = librosa.resample(
                y=np.array(audio_data, dtype=np.float32), 
                orig_sr=sr, 
                target_sr=self.audio_sample_rate
            )
        
        # 2. Numpy -> Tensor
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        # 3. 정규화 (AudioPreprocessor 사용)
        # 중요: AudioPreprocessor는 __call__이 없으므로 메서드를 직접 호출
        waveform = self.preprocessor.normalize_audio(waveform)
        
        return waveform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 1. 오디오 데이터 추출
        # HuggingFace Dataset의 audio 컬럼은 {'array': ..., 'sampling_rate': ...} 형태
        audio_info = sample["audio"]
        raw_audio = audio_info["array"]
        orig_sr = audio_info["sampling_rate"]
        
        # 2. 웨이브폼 처리 (리샘플링 + 정규화)
        waveform = self._process_waveform(raw_audio, orig_sr)
        
        # 3. 레이블 파싱
        config_name = sample["config_name"]  # 예: 'irish_male'
        region_id, gender_id, coords = self._parse_label(config_name)
        
        return {
            "waveform": waveform,      # (T,) 1D Tensor
            "region_id": region_id,
            "gender_id": gender_id,
            "coords": coords,
            "config_name": config_name
        }


def collate_fn(batch):
    """
    DataLoader용 collate function
    - 가변 길이 오디오 Padding (Zero Padding)
    - Attention Mask 생성
    - 레이블 Stacking
    """
    # 1. Waveform Padding 처리
    waveforms = [b["waveform"] for b in batch]
    
    # 배치 내 최대 길이 계산
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    batch_size = len(batch)
    
    # 텐서 초기화 (Padding 값은 0.0)
    input_values = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, wave in enumerate(waveforms):
        length = wave.shape[0]
        input_values[i, :length] = wave
        attention_mask[i, :length] = 1  # 실제 데이터가 있는 곳은 1, 패딩은 0
    
    # 2. 레이블 텐서 변환
    region_labels = torch.tensor([b["region_id"] for b in batch], dtype=torch.long)
    gender_labels = torch.tensor([b["gender_id"] for b in batch], dtype=torch.long)
    
    # 3. 좌표 텐서 변환 (B, 2)
    coords = torch.tensor([b["coords"] for b in batch], dtype=torch.float32)
    
    return {
        "input_values": input_values,      # (B, T_max) -> 모델 입력
        "attention_mask": attention_mask,  # (B, T_max) -> 모델 입력
        "region_labels": region_labels,    # (B,)
        "gender_labels": gender_labels,    # (B,)
        "coords": coords                   # (B, 2)
    }