"""
Custom PyTorch Dataset 구현
"""
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from data.data_config import (
    DATASET_NAME, REGION_LABELS, GENDER_LABELS,
    REGION_COORDS, normalize_coords, AUDIO_SAMPLE_RATE
)
from data.preprocessing import AudioPreprocessor

class EnglishDialectsDataset(Dataset):
    """
    English Dialects 데이터셋
    
    레이블 형식: 'irish_male', 'irish_female', 'midlands_male', ... 등
    우리는 메인 레이블: 지역 (6개)
         보조 레이블: 성별 (2개)
    """
    
    def __init__(self, split='train', use_augment=False, processor=None, 
                data_dir="./data/english_dialects"):
        super().__init__()
        self.split = split
        self.use_augment = use_augment
        self.processor = processor
        
        # 로컬에서 로드
        from datasets import load_from_disk
        local_path = f"{data_dir}/{split}"
        
        if os.path.exists(local_path):
            self.dataset = load_from_disk(local_path)
            print(f"✅ Loaded from disk: {len(self.dataset)} samples")
        else:
            raise FileNotFoundError(
                f"Dataset not found at {local_path}. "
                f"Please run 'python download_dataset.py' first."
            )

        # 4. 전처리기 초기화
        #self.dataset = None
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=AUDIO_SAMPLE_RATE,
            use_augment=use_augment
        )
        self.processor = processor
    
    def __len__(self):
        '''전체 데이터셋의 샘플 개수 반환'''
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # 1. 이미 디코딩된 오디오 가져오기
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]

        # 2. 정규화
        waveform = self.audio_preprocessor.normalize_audio(waveform)

        # 3. 레이블 파싱
        config_name = sample["config_name"]  # 예: 'irish_male'
        region, gender, region_id, gender_id, coords = self._parse_label(config_name)

        return {
            "waveform": waveform,
            "sample_rate": sr,
            "region_id": region_id,
            "gender_id": gender_id,
            "coords": coords,
            "config_name": config_name #1
        }

   
def collate_fn(batch):
    """
    DataLoader용 collate function
    - 가변 길이 오디오를 padding
    - 레이블은 그대로 텐서로 변환
    """
    import torch

    # 1. 배치에서 input_values 추출 및 padding (waveform은 (1, T) 형태)
    lengths = [b["waveform"].shape[-1] for b in batch]
    max_len = int(max(lengths))
    bs = len(batch)

    input_values = torch.zeros(bs, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        w = b["waveform"].squeeze(0)  # (T,)
        L = w.shape[0]
        input_values[i, :L] = w
        attention_mask[i, :L] = 1    
    
    # 2) 라벨 스택
    region_labels = torch.tensor([b["region_id"] for b in batch], dtype=torch.long)
    gender_labels = torch.tensor([b["gender_id"] for b in batch], dtype=torch.long)

    # 3) 좌표 스택 (nlat, nlon) -> (B, 2)
    coords = torch.tensor([b["coords"] for b in batch], dtype=torch.float32)

    # 4) dict로 반환
    return {
        "input_values": input_values,        # (B, T_max)
        "attention_mask": attention_mask,    # (B, T_max)
        "region_labels": region_labels,      # (B,)
        "gender_labels": gender_labels,      # (B,)
        "coords": coords                     # (B, 2)
    }

