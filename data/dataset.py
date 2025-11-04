# ============================================================================
# 👤 PERSON A: 파일 3: data/dataset.py
# ============================================================================

"""
Custom PyTorch Dataset 구현
"""

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
    
    def __init__(self, split='train', use_augment=False, processor=None):
        """
        Args:
            split: 'train', 'validation', 'test'
            use_augment: SpecAugment 사용 여부
            processor: Wav2Vec2Processor 인스턴스
        """
        # TODO: 구현
        # 1. HuggingFace datasets 라이브러리로 데이터셋 로드
        # 2. split별로 데이터 필터링
        # 3. 레이블 파싱 (예: 'irish_male' -> region='irish', gender='male')
        # 4. 전처리기 초기화
        self.dataset = None
        self.preprocessor = AudioPreprocessor(
            sample_rate=AUDIO_SAMPLE_RATE,
            use_augment=use_augment
        )
        self.processor = processor
    
    def __len__(self):
        """데이터셋 크기"""
        # TODO: 구현
        pass
    
    def __getitem__(self, idx):
        """
        Args:
            idx: 샘플 인덱스
        
        Returns:
            dict: {
                'audio': processed audio,
                'region_label': 지역 레이블 (0-5),
                'gender_label': 성별 레이블 (0-1),
                'region_coords': 정규화된 위도/경도
            }
        """
        # TODO: 구현
        # 1. self.dataset[idx] 접근
        # 2. 오디오 파일 경로에서 오디오 로드
        # 3. 레이블 파싱 (예: 'irish_male' -> region, gender)
        # 4. 좌표 가져오기 및 정규화
        # 5. Wav2Vec2Processor로 처리 (input_values 반환)
        # 6. dict 형태로 반환
        pass

def collate_fn(batch):
    """
    DataLoader용 collate function
    - 가변 길이 오디오를 padding
    - 레이블은 그대로 텐서로 변환
    
    Args:
        batch: 샘플 리스트
    
    Returns:
        dict: {
            'input_values': (batch_size, max_length),
            'attention_mask': (batch_size, max_length),
            'region_labels': (batch_size,),
            'gender_labels': (batch_size,),
            'coords': (batch_size, 2)
        }
    """
    # TODO: 구현
    # 1. 배치에서 input_values 추출 및 padding
    # 2. attention_mask 생성
    # 3. 레이블 스택
    # 4. 좌표 스택
    # 5. dict로 반환
    pass
