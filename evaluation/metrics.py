"""
Metrics Calculation Module
순수 메트릭 계산 함수들
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall accuracy 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
    
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def calculate_per_class_accuracy(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Per-class accuracy 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
    
    Returns:
        Dictionary of per-class accuracies
    """
    classes = np.unique(y_true)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in classes]
    
    per_class_acc = {}
    for idx, class_label in enumerate(classes):
        mask = y_true == class_label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_acc[class_names[idx]] = class_acc
    
    return per_class_acc


def calculate_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro F1 score 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
    
    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def calculate_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted F1 score 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
    
    Returns:
        Weighted F1 score
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def calculate_per_class_f1(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Per-class F1 score 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
    
    Returns:
        Dictionary of per-class F1 scores
    """
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    if class_names is None:
        classes = np.unique(y_true)
        class_names = [f"Class {i}" for i in classes]
    
    return {class_names[idx]: score for idx, score in enumerate(f1_scores)}


def calculate_precision_recall(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               average: str = 'macro') -> Dict[str, float]:
    """
    Precision과 Recall 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        average: 'macro', 'weighted', 'micro', None
    
    Returns:
        Dictionary with precision and recall
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix 생성
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
    
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              output_dict: bool = False):
    """
    Classification report 생성
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        output_dict: True면 dict, False면 string
    
    Returns:
        Classification report
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def calculate_all_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None) -> Dict:
    """
    모든 메트릭을 한 번에 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
    
    Returns:
        Dictionary containing all metrics
    """
    return {
        'overall_accuracy': calculate_accuracy(y_true, y_pred),
        'per_class_accuracy': calculate_per_class_accuracy(y_true, y_pred, class_names),
        'f1_macro': calculate_f1_macro(y_true, y_pred),
        'f1_weighted': calculate_f1_weighted(y_true, y_pred),
        'per_class_f1': calculate_per_class_f1(y_true, y_pred, class_names),
        'precision_recall': calculate_precision_recall(y_true, y_pred),
        'confusion_matrix': get_confusion_matrix(y_true, y_pred),
        'classification_report': get_classification_report(y_true, y_pred, class_names, output_dict=True)
    }


# 사용 예제
if __name__ == "__main__":
    # 예제 데이터
    np.random.seed(42)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 1, 2, 0, 2, 2])
    class_names = ['Cat', 'Dog', 'Bird']
    
    # 개별 메트릭 계산
    print("Overall Accuracy:", calculate_accuracy(y_true, y_pred))
    print("F1 Macro:", calculate_f1_macro(y_true, y_pred))
    print("Per-class Accuracy:", calculate_per_class_accuracy(y_true, y_pred, class_names))
    
    # 전체 메트릭 계산
    all_metrics = calculate_all_metrics(y_true, y_pred, class_names)
    print("\nAll Metrics:", all_metrics)

class AccentMetrics:
    """Stateless wrapper for accent metric calculations."""

    @staticmethod
    def accuracy(y_true, y_pred):
        return calculate_accuracy(y_true, y_pred)

    @staticmethod
    def f1_macro(y_true, y_pred):
        return calculate_f1_macro(y_true, y_pred)

    @staticmethod
    def f1_weighted(y_true, y_pred):
        return calculate_f1_weighted(y_true, y_pred)

    @staticmethod
    def per_class_accuracy(y_true, y_pred, class_names=None):
        return calculate_per_class_accuracy(y_true, y_pred, class_names)

    @staticmethod
    def per_class_f1(y_true, y_pred, class_names=None):
        return calculate_per_class_f1(y_true, y_pred, class_names)

    @staticmethod
    def precision_recall(y_true, y_pred, average='macro'):
        return calculate_precision_recall(y_true, y_pred, average)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return get_confusion_matrix(y_true, y_pred)

    @staticmethod
    def classification_report(y_true, y_pred, class_names=None, output_dict=False):
        return get_classification_report(y_true, y_pred, class_names, output_dict)

    @staticmethod
    def all_metrics(y_true, y_pred, class_names=None):
        return calculate_all_metrics(y_true, y_pred, class_names)
