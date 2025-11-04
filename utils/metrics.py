# ============================================================================
# 👤 PERSON C: 평가 및 실험 관리
# 파일 1: metrics.py
# ============================================================================

"""
평가 메트릭 계산
- Accuracy (overall, per-class)
- F1-score (macro, weighted)
- Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch

def compute_metrics(predictions, labels, label_type='region'):
    """
    메트릭 계산
    
    Args:
        predictions: (batch_size,) 또는 (batch_size, num_classes) 
                    - 후자면 argmax 취함
        labels: (batch_size,) 정수 레이블
        label_type: 'region' 또는 'gender'
    
    Returns:
        dict: {
            'accuracy': float,
            'f1_macro': float,
            'f1_weighted': float,
            'confusion_matrix': np.array,
            'classification_report': str
        }
    """
    # TODO: 구현
    # 1. predictions이 logits이면 argmax 취하기
    # 2. numpy로 변환
    # 3. accuracy_score 계산
    # 4. f1_score 계산 (macro, weighted)
    # 5. confusion_matrix 계산
    # 6. classification_report 생성
    # 7. dict 형태 반환
    pass

def compute_metrics_per_class(predictions, labels, class_names=None):
    """
    클래스별 메트릭
    
    Args:
        predictions: (batch_size,)
        labels: (batch_size,)
        class_names: 클래스 이름 리스트 (예: ['irish', 'midlands', ...])
    
    Returns:
        dict: {
            'class_name': {
                'accuracy': float,
                'f1': float,
                'precision': float,
                'recall': float
            },
            ...
        }
    """
    # TODO: 구현
    pass