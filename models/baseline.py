# ============================================================================
# 👤 PERSON B: 베이스라인 모델 리드
# 파일 1: models/baseline.py
# ============================================================================

"""
베이스라인 모델: Wav2Vec2 + Classification Head
- Multi-task learning (지역 분류 + 성별 분류)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2Baseline(nn.Module):
    """
    Wav2Vec2 기반 억양 분류 모델
    
    구조:
    - Wav2Vec2Model (사전학습 모델)
    - Temporal pooling (평균)
    - Classification head (지역 + 성별 분류)
    """
    
    def __init__(self, model_name="facebook/wav2vec2-xls-r-300m", num_regions=6, num_genders=2):
        """
        Args:
            model_name: HuggingFace pretrained 모델 이름
            num_regions: 지역 클래스 수 (기본 6)
            num_genders: 성별 클래스 수 (기본 2)
        """
        super().__init__()
        
        # Wav2Vec2 모델 로드
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size  # 보통 1024 or 768
        
        # Classification head
        # TODO: 구현
        # 1. Dropout layer
        # 2. Region classifier (linear layer: hidden_size -> num_regions)
        # 3. Gender classifier (linear layer: hidden_size -> num_genders)
        
        self.num_regions = num_regions
        self.num_genders = num_genders
    
    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: (batch_size, seq_length) - 오디오 파형
            attention_mask: (batch_size, seq_length) - padding mask
        
        Returns:
            dict: {
                'region_logits': (batch_size, num_regions),
                'gender_logits': (batch_size, num_genders),
                'pooled_hidden': (batch_size, hidden_size) - 시각화용
            }
        """
        # TODO: 구현
        # 1. Wav2Vec2로 feature 추출
        #    outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        #    last_hidden = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        #
        # 2. Temporal pooling (평균)
        #    if attention_mask is not None:
        #        # mask된 부분 제외하고 평균
        #    else:
        #        pooled = last_hidden.mean(dim=1)  # (batch_size, hidden_size)
        #
        # 3. Classification
        #    region_logits = self.region_classifier(pooled)
        #    gender_logits = self.gender_classifier(pooled)
        #
        # 4. 반환
        pass