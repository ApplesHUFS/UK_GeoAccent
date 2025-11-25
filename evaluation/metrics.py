"""
Metrics Calculation Module
Contains pure metric computation functions.
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
    """Compute overall accuracy."""
    return accuracy_score(y_true, y_pred)


def calculate_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute per-class accuracy."""
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
    """Compute macro F1 score."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def calculate_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute weighted F1 score."""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def calculate_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute per-class F1 scores."""
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)

    if class_names is None:
        classes = np.unique(y_true)
        class_names = [f"Class {i}" for i in classes]

    return {class_names[idx]: score for idx, score in enumerate(f1_scores)}


def calculate_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """Compute precision and recall."""
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    return {
        'precision': precision,
        'recall': recall
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Generate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dict: bool = False
):
    """Generate classification report."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """Compute all metrics at once."""
    return {
        'overall_accuracy': calculate_accuracy(y_true, y_pred),
        'per_class_accuracy': calculate_per_class_accuracy(y_true, y_pred, class_names),
        'f1_macro': calculate_f1_macro(y_true, y_pred),
        'f1_weighted': calculate_f1_weighted(y_true, y_pred),
        'per_class_f1': calculate_per_class_f1(y_true, y_pred, class_names),
        'precision_recall': calculate_precision_recall(y_true, y_pred),
        'confusion_matrix': get_confusion_matrix(y_true, y_pred),
        'classification_report': get_classification_report(
            y_true, y_pred, class_names, output_dict=True
        )
    }


class AccentMetrics:
    """Stateless wrapper for accent-related metric calculations."""

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
