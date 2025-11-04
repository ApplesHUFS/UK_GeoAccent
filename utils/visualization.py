# ============================================================================
# 👤 PERSON D: 문서화 및 시각화
# 파일 1: visualization.py
# ============================================================================

"""
시각화 함수
- 학습 곡선
- Confusion matrix
- Per-class metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

def plot_confusion_matrix(predictions, labels, class_names, title="Confusion Matrix", save_path=None):
    """
    Confusion matrix 시각화
    
    Args:
        predictions: (n_samples,) 예측 레이블
        labels: (n_samples,) 실제 레이블
        class_names: 클래스 이름 리스트
        title: 그래프 제목
        save_path: 저장 경로 (None이면 저장 안 함)
    """
    # TODO: 구현
    # 1. confusion_matrix 계산
    # 2. Heatmap으로 시각화
    # 3. 저장 (if save_path is not None)
    pass

def plot_training_curves(train_losses, val_losses, val_accuracies, save_path=None):
    """
    학습 곡선 시각화
    
    Args:
        train_losses: 에포크별 훈련 손실값 리스트
        val_losses: 에포크별 검증 손실값 리스트
        val_accuracies: 에포크별 검증 정확도 리스트
        save_path: 저장 경로
    """
    # TODO: 구현
    # 1. 3개의 서브플롯 생성
    #    - Train vs Val loss
    #    - Val accuracy
    #    - 함께
    # 2. 저장
    pass

def plot_per_class_metrics(metrics_dict, class_names, save_path=None):
    """
    클래스별 메트릭 시각화 (F1, Precision, Recall)
    
    Args:
        metrics_dict: 클래스별 메트릭 딕셔너리
                     {
                        'class_name': {'f1': float, 'precision': float, 'recall': float},
                        ...
                     }
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
    """
    # TODO: 구현
    # 1. Bar plot으로 각 메트릭 시각화
    # 2. 저장
    pass

def plot_waveform(audio_path, title="Waveform", save_path=None):
    """
    오디오 파형 시각화
    
    Args:
        audio_path: 오디오 파일 경로
        title: 제목
        save_path: 저장 경로
    """
    # TODO: 구현
    # 1. 오디오 로드
    # 2. 파형 시각화
    # 3. 저장
    pass