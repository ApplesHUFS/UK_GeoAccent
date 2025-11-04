# ============================================================================
# 👤 PERSON C: 파일 2: evaluate.py
# ============================================================================

"""
학습된 모델 평가
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from models.baseline import Wav2Vec2Baseline
from data.dataset import EnglishDialectsDataset, collate_fn
from data.data_config import ID_TO_REGION, ID_TO_GENDER
from metrics import compute_metrics, compute_metrics_per_class

class Evaluator:
    """평가 클래스"""
    
    def __init__(self, checkpoint_path, config):
        """
        Args:
            checkpoint_path: 저장된 모델 경로
            config: 설정 딕셔너리
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = Wav2Vec2Baseline(
            model_name=config['model_name'],
            num_regions=6,
            num_genders=2
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 데이터 로더
        self.test_loader = DataLoader(
            EnglishDialectsDataset(split='test', use_augment=False),
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            shuffle=False
        )
    
    def evaluate(self):
        """
        Test set 평가
        
        Returns:
            dict: {
                'region_metrics': {...},
                'gender_metrics': {...},
                'predictions': [...],
                'labels': [...]
            }
        """
        # TODO: 구현
        # 1. 모든 배치에 대해 예측
        # 2. 예측값 저장
        # 3. region과 gender 메트릭 각각 계산
        # 4. 결과 반환
        
        all_region_preds = []
        all_gender_preds = []
        all_region_labels = []
        all_gender_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # TODO: 구현
                # input_values = batch['input_values'].to(self.device)
                # attention_mask = batch['attention_mask'].to(self.device)
                # region_labels = batch['region_labels']
                # gender_labels = batch['gender_labels']
                #
                # outputs = self.model(input_values, attention_mask)
                # region_preds = outputs['region_logits'].argmax(dim=1).cpu()
                # gender_preds = outputs['gender_logits'].argmax(dim=1).cpu()
                #
                # all_region_preds.extend(region_preds)
                # all_gender_preds.extend(gender_preds)
                # all_region_labels.extend(region_labels)
                # all_gender_labels.extend(gender_labels)
                pass
        
        # 메트릭 계산
        region_metrics = compute_metrics(
            all_region_preds, all_region_labels, label_type='region'
        )
        gender_metrics = compute_metrics(
            all_gender_preds, all_gender_labels, label_type='gender'
        )
        
        return {
            'region_metrics': region_metrics,
            'gender_metrics': gender_metrics,
            'region_predictions': all_region_preds,
            'gender_predictions': all_gender_preds,
            'region_labels': all_region_labels,
            'gender_labels': all_gender_labels
        }

def save_results(results, output_path):
    """
    평가 결과 저장
    
    Args:
        results: 평가 결과 딕셔너리
        output_path: 저장 경로
    """
    # TODO: 구현
    # JSON 형식으로 저장 (numpy array는 tolist() 필요)
    pass
