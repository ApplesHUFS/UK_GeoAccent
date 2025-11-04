# ============================================================================
# 👤 PERSON B: 파일 2: train.py
# ============================================================================

"""
학습 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
import yaml
from pathlib import Path

# Local imports (각각의 담당자 코드)
from models.baseline import Wav2Vec2Baseline
from data.dataset import EnglishDialectsDataset, collate_fn
from metrics import compute_metrics  # Person C가 구현할 것

class Trainer:
    """모델 학습 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: 설정 딕셔너리 (YAML에서 로드)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Processor 로드
        self.processor = Wav2Vec2Processor.from_pretrained(
            config['model_name']
        )
        
        # 모델 초기화
        self.model = Wav2Vec2Baseline(
            model_name=config['model_name'],
            num_regions=6,
            num_genders=2
        ).to(self.device)
        
        # Loss functions
        self.region_loss_fn = nn.CrossEntropyLoss()
        self.gender_loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # 데이터 로더
        self.train_loader = self._get_dataloader('train')
        self.val_loader = self._get_dataloader('validation')
    
    def _get_dataloader(self, split):
        """
        DataLoader 생성
        
        Args:
            split: 'train', 'validation', 'test'
        
        Returns:
            DataLoader
        """
        # TODO: 구현
        # 1. EnglishDialectsDataset 초기화 (split, augment 설정)
        # 2. DataLoader 생성 (collate_fn 포함)
        # 3. 반환
        pass
    
    def train_epoch(self):
        """한 에포크 학습"""
        # TODO: 구현
        # 1. self.model.train()
        # 2. train_loader 순회
        # 3. Forward pass
        #    outputs = self.model(input_values, attention_mask)
        #    region_logits = outputs['region_logits']
        #    gender_logits = outputs['gender_logits']
        #
        # 4. Loss 계산 (weighted combination)
        #    region_loss = self.region_loss_fn(region_logits, region_labels)
        #    gender_loss = self.gender_loss_fn(gender_logits, gender_labels)
        #    total_loss = 0.8 * region_loss + 0.2 * gender_loss
        #
        # 5. Backward + optimizer step
        #
        # 6. Loss 추적 (평균값 반환)
        pass
    
    def validate(self):
        """Validation"""
        # TODO: 구현
        # 1. self.model.eval()
        # 2. torch.no_grad() context
        # 3. val_loader 순회
        # 4. 예측값 저장
        # 5. compute_metrics() 호출하여 정확도, F1 등 계산
        # 6. 반환
        pass
    
    def train(self, num_epochs):
        """전체 학습 루프"""
        # TODO: 구현
        # 1. num_epochs 만큼 반복
        # 2. train_epoch() 호출
        # 3. validate() 호출
        # 4. Early stopping 구현
        # 5. Checkpoint 저장 (best 모델)
        # 6. 로깅
        pass

def main():
    """Main 함수"""
    # TODO: 구현
    # 1. configs/experiment_config.yaml 로드
    # 2. Trainer 초기화
    # 3. trainer.train() 호출
    # 4. 결과 저장
    
    # 예시 코드 (수정 필요)
    config_path = Path('configs/experiment_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train(num_epochs=config['num_epochs'])

if __name__ == '__main__':
    main()
