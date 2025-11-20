"""
Model Evaluation Module
모델 평가 워크플로우 및 시각화
UK GeoAccent 프로젝트용 완전한 평가 스크립트
"""


import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union

# metrics 모듈에서 함수 import
from metrics import (
    calculate_accuracy,
    calculate_per_class_accuracy,
    calculate_f1_macro,
    calculate_f1_weighted,
    calculate_per_class_f1,
    calculate_precision_recall,
    get_confusion_matrix,
    get_classification_report,
    calculate_all_metrics
)


class ModelEvaluator:
    """모델 평가를 위한 종합 클래스"""
    
    def __init__(self, 
                 y_true: Optional[np.ndarray] = None, 
                 y_pred: Optional[np.ndarray] = None, 
                 class_names: Optional[List[str]] = None,
                 model: Optional[nn.Module] = None,
                 test_loader: Optional[DataLoader] = None,
                 device: str = 'cuda'):
        """
        Args:
            y_true: 실제 레이블 (옵션)
            y_pred: 예측 레이블 (옵션)
            class_names: 클래스 이름 리스트
            model: PyTorch 모델 (옵션)
            test_loader: 테스트 데이터 로더 (옵션)
            device: 'cuda' 또는 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.test_loader = test_loader
        
        # 모델이 제공된 경우 평가 모드로 설정
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)
        
        # 레이블 설정
        if y_true is not None and y_pred is not None:
            self.y_true = np.array(y_true)
            self.y_pred = np.array(y_pred)
        else:
            self.y_true = None
            self.y_pred = None
        
        # 클래스 이름 설정
        if self.y_true is not None:
            self.classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
            if class_names is None:
                self.class_names = [f"Class {i}" for i in self.classes]
            else:
                self.class_names = class_names
        else:
            self.class_names = class_names if class_names else []
        
        # 메트릭 저장
        self.metrics = None
        self.y_proba = None
    
    def predict_from_loader(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 로더로부터 예측 수행
        
        Args:
            verbose: 진행 상황 표시 여부
        
        Returns:
            (y_true, y_pred, y_proba) tuple
        """
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and test_loader must be provided for prediction")
        
        if verbose:
            print("\n" + "="*70)
            print("🔮 Running Predictions...")
            print("="*70)
        
        all_labels = []
        all_preds = []
        all_probas = []
        
        with torch.no_grad():
            iterator = tqdm(self.test_loader, desc="Predicting") if verbose else self.test_loader
            
            for batch in iterator:
                # 입력 데이터 추출
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
            
                # 레이블 선택
                if task == 'region':
                    labels = batch['region_labels'].to(self.device)
                else:  # gender
                    labels = batch['gender_labels'].to(self.device)
                
                # 예측
                outputs = self.model(input_values)
                
                # 출력 처리
                if outputs.dim() == 1 or outputs.shape[1] == 1:
                    # Binary classification
                    probas = torch.sigmoid(outputs.squeeze())
                    preds = (probas > 0.5).long()
                    probas = torch.stack([1-probas, probas], dim=1)
                else:
                    # Multi-class classification
                    probas = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
            
            # 결과 저장
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
        
        # 결합
        self.y_true = np.concatenate(all_labels)
        self.y_pred = np.concatenate(all_preds)
        self.y_proba = np.concatenate(all_probas)
        
        if verbose:
            print(f"✅ Predictions completed: {len(self.y_true)} samples")
        
        return self.y_true, self.y_pred, self.y_proba
    
    def calculate_metrics(self) -> Dict:
        """모든 메트릭 계산"""
        if self.y_true is None or self.y_pred is None:
            if self.model is not None and self.test_loader is not None:
                self.predict_from_loader()
            else:
                raise ValueError("No predictions available. Provide y_true/y_pred or model/test_loader")
        
        print("\n" + "="*70)
        print("📊 Calculating Metrics...")
        print("="*70)
        
        self.metrics = calculate_all_metrics(
            self.y_true, 
            self.y_pred, 
            self.class_names
        )
        
        return self.metrics
    
    def print_summary(self):
        """전체 메트릭 요약 출력"""
        if self.metrics is None:
            self.calculate_metrics()
        
        print("\n" + "="*70)
        print("📈 MODEL EVALUATION SUMMARY")
        print("="*70)
        
        # Overall Accuracy
        print(f"\n📊 Overall Accuracy: {self.metrics['overall_accuracy']:.4f} ({self.metrics['overall_accuracy']*100:.2f}%)")
        
        # Per-Class Accuracy
        print("\n📋 Per-Class Accuracy:")
        for class_name, acc in self.metrics['per_class_accuracy'].items():
            support = np.sum(self.y_true == self.class_names.index(class_name))
            print(f"  • {class_name:20s}: {acc:.4f} ({acc*100:.2f}%) | Support: {support}")
        
        # F1 Scores
        print(f"\n📈 F1 Score (Macro):    {self.metrics['f1_macro']:.4f}")
        print(f"📈 F1 Score (Weighted): {self.metrics['f1_weighted']:.4f}")
        
        # Per-Class F1
        print("\n📋 Per-Class F1 Score:")
        for class_name, f1 in self.metrics['per_class_f1'].items():
            print(f"  • {class_name:20s}: {f1:.4f}")
        
        # Precision & Recall
        pr = self.metrics['precision_recall']
        print(f"\n🎯 Precision (Macro):   {pr['precision']:.4f}")
        print(f"🎯 Recall (Macro):      {pr['recall']:.4f}")
    
    def print_classification_report(self):
        """Classification report 출력"""
        print("\n" + "="*70)
        print("📋 DETAILED CLASSIFICATION REPORT")
        print("="*70)
        report = get_classification_report(
            self.y_true, 
            self.y_pred, 
            self.class_names, 
            output_dict=False
        )
        print(report)
    
    def plot_confusion_matrix(self, 
                             figsize: Tuple[int, int] = (10, 8),
                             cmap: str = 'Blues',
                             save_path: Optional[str] = None,
                             show_percentages: bool = False,
                             normalize: bool = False) -> plt.Figure:
        """
        Confusion matrix 시각화
        
        Args:
            figsize: Figure 크기
            cmap: Color map
            save_path: 저장 경로
            show_percentages: 백분율 표시 여부
            normalize: 정규화 여부
        
        Returns:
            Figure object
        """
        if self.metrics is None:
            self.calculate_metrics()
        
        cm = self.metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        elif show_percentages:
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            annot = np.array([[f'{int(count)}\n({percent:.1f}%)' 
                              for count, percent in zip(row_counts, row_percents)]
                             for row_counts, row_percents in zip(cm, cm_percent)])
            sns.heatmap(cm, annot=annot, fmt='', cmap=cmap,
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Count'},
                       ax=ax)
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Confusion matrix saved to {save_path}")
            
            return fig
        else:
            fmt = 'd'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Normalized' if normalize else 'Count'},
                   ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, 
                               figsize: Tuple[int, int] = (14, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        클래스별 메트릭 비교 시각화
        
        Args:
            figsize: Figure 크기
            save_path: 저장 경로
        
        Returns:
            Figure object
        """
        if self.metrics is None:
            self.calculate_metrics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Per-class Accuracy
        classes = list(self.metrics['per_class_accuracy'].keys())
        accuracies = list(self.metrics['per_class_accuracy'].values())
        
        bars1 = ax1.bar(range(len(classes)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.set_xlabel('Class', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
        ax1.set_title('Per-Class Accuracy', fontweight='bold', pad=10, fontsize=13)
        ax1.set_ylim([0, 1.05])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=self.metrics['overall_accuracy'], color='red', linestyle='--', 
                   label=f"Overall: {self.metrics['overall_accuracy']:.3f}", linewidth=2)
        ax1.legend()
        
        # 값 표시
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Per-class F1 Score
        f1_scores = list(self.metrics['per_class_f1'].values())
        
        bars2 = ax2.bar(range(len(classes)), f1_scores, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.set_xlabel('Class', fontweight='bold', fontsize=11)
        ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=11)
        ax2.set_title('Per-Class F1 Score', fontweight='bold', pad=10, fontsize=13)
        ax2.set_ylim([0, 1.05])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.axhline(y=self.metrics['f1_macro'], color='red', linestyle='--', 
                   label=f"Macro: {self.metrics['f1_macro']:.3f}", linewidth=2)
        ax2.legend()
        
        # 값 표시
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Metrics comparison saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, 
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        ROC Curve 시각화 (multi-class)
        
        Args:
            figsize: Figure 크기
            save_path: 저장 경로
        
        Returns:
            Figure object
        """
        if self.y_proba is None:
            print("⚠️ Probability scores not available. Skipping ROC curve.")
            return None
        
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # One-hot encoding
        y_true_bin = label_binarize(self.y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 각 클래스별 ROC 곡선
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, 
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # 대각선 (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curves - Multi-Class', fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curves saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        메트릭을 DataFrame으로 정리
        
        Args:
            output_path: CSV 저장 경로
        
        Returns:
            DataFrame containing metrics
        """
        if self.metrics is None:
            self.calculate_metrics()
        
        report_data = []
        
        # Per-class metrics
        for idx, class_name in enumerate(self.class_names):
            support = np.sum(self.y_true == idx)
            report_data.append({
                'Class': class_name,
                'Accuracy': self.metrics['per_class_accuracy'].get(class_name, 0),
                'F1-Score': self.metrics['per_class_f1'].get(class_name, 0),
                'Support': support
            })
        
        df = pd.DataFrame(report_data)
        
        # Overall metrics
        overall_row = pd.DataFrame([{
            'Class': 'Overall',
            'Accuracy': self.metrics['overall_accuracy'],
            'F1-Score': self.metrics['f1_macro'],
            'Support': len(self.y_true)
        }])
        
        df = pd.concat([df, overall_row], ignore_index=True)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Report saved to {output_path}")
        
        return df
    
    def save_results(self, save_dir: Union[str, Path]):
        """
        모든 결과를 저장
        
        Args:
            save_dir: 저장 디렉토리
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {save_dir}...")
        
        # 메트릭 JSON 저장
        metrics_path = save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            # numpy array를 list로 변환
            metrics_serializable = {
                'overall_accuracy': float(self.metrics['overall_accuracy']),
                'f1_macro': float(self.metrics['f1_macro']),
                'f1_weighted': float(self.metrics['f1_weighted']),
                'per_class_accuracy': {k: float(v) for k, v in self.metrics['per_class_accuracy'].items()},
                'per_class_f1': {k: float(v) for k, v in self.metrics['per_class_f1'].items()},
                'precision_recall': {k: float(v) for k, v in self.metrics['precision_recall'].items()},
                'confusion_matrix': self.metrics['confusion_matrix'].tolist(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            json.dump(metrics_serializable, f, indent=4)
        print(f"  ✅ Metrics saved to {metrics_path}")
        
        # Predictions 저장
        predictions_path = save_dir / 'predictions.npz'
        np.savez(predictions_path, 
                y_true=self.y_true, 
                y_pred=self.y_pred,
                y_proba=self.y_proba if self.y_proba is not None else np.array([]))
        print(f"  ✅ Predictions saved to {predictions_path}")
        
        # CSV report 저장
        report_path = save_dir / 'evaluation_report.csv'
        self.generate_report(output_path=str(report_path))
        
        # Classification report 저장
        clf_report_path = save_dir / 'classification_report.txt'
        with open(clf_report_path, 'w') as f:
            report = get_classification_report(
                self.y_true, 
                self.y_pred, 
                self.class_names, 
                output_dict=False
            )
            f.write(report)
        print(f"  ✅ Classification report saved to {clf_report_path}")
    
    def full_evaluation(self, 
                       save_dir: Optional[Union[str, Path]] = None,
                       show_plots: bool = True,
                       save_plots: bool = True) -> Dict:
        """
        전체 평가 수행 (예측 + 메트릭 계산 + 시각화 + 저장)
        
        Args:
            save_dir: 저장 디렉토리
            show_plots: 플롯 표시 여부
            save_plots: 플롯 저장 여부
        
        Returns:
            Dictionary containing all metrics
        """
        # 1. 예측 (필요한 경우)
        if self.y_true is None and self.model is not None:
            self.predict_from_loader()
        
        # 2. 메트릭 계산
        self.calculate_metrics()
        
        # 3. 결과 출력
        self.print_summary()
        self.print_classification_report()
        
        # 4. 시각화
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        cm_path = save_dir / 'confusion_matrix.png' if save_plots and save_dir else None
        self.plot_confusion_matrix(save_path=cm_path, show_percentages=True)
        
        # Metrics Comparison
        metrics_path = save_dir / 'metrics_comparison.png' if save_plots and save_dir else None
        self.plot_metrics_comparison(save_path=metrics_path)
        
        # ROC Curves
        if self.y_proba is not None:
            roc_path = save_dir / 'roc_curves.png' if save_plots and save_dir else None
            self.plot_roc_curves(save_path=roc_path)
        
        # 5. 결과 저장
        if save_dir:
            self.save_results(save_dir)
        
        # 6. DataFrame 출력
        df = self.generate_report()
        print("\n" + "="*70)
        print("📊 METRICS TABLE")
        print("="*70)
        print(df.to_string(index=False))
        
        # 7. 플롯 표시
        if show_plots:
            plt.show()
        else:
            plt.close('all')
        
        return self.metrics


def evaluate_model(y_true: Optional[np.ndarray] = None,
                  y_pred: Optional[np.ndarray] = None,
                  class_names: Optional[List[str]] = None,
                  model: Optional[nn.Module] = None,
                  test_loader: Optional[DataLoader] = None,
                  device: str = 'cuda',
                  save_dir: Optional[str] = None,
                  show_plots: bool = True) -> ModelEvaluator:
    """
    간편한 모델 평가 함수
    
    Args:
        y_true: 실제 레이블 (옵션)
        y_pred: 예측 레이블 (옵션)
        class_names: 클래스 이름
        model: PyTorch 모델 (옵션)
        test_loader: 테스트 데이터 로더 (옵션)
        device: 'cuda' 또는 'cpu'
        save_dir: 저장 디렉토리
        show_plots: 플롯 표시 여부
    
    Returns:
        ModelEvaluator instance
    """
    evaluator = ModelEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # 전체 평가 수행
    evaluator.full_evaluation(
        save_dir=save_dir,
        show_plots=show_plots,
        save_plots=True
    )
    
    return evaluator


def load_model_and_evaluate(model_path: str,
                           model_class: nn.Module,
                           test_loader: DataLoader,
                           class_names: List[str],
                           device: str = 'cuda',
                           save_dir: Optional[str] = None) -> ModelEvaluator:
    """
    저장된 모델을 불러와서 평가
    
    Args:
        model_path: 모델 체크포인트 경로
        model_class: 모델 클래스
        test_loader: 테스트 데이터 로더
        class_names: 클래스 이름
        device: 'cuda' 또는 'cpu'
        save_dir: 저장 디렉토리
    
    Returns:
        ModelEvaluator instance
    """
    print(f"\n📥 Loading model from {model_path}...")
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded")
    
    # 평가
    return evaluate_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        save_dir=save_dir
    )


# 사용 예제
if __name__ == "__main__":
    # 예제 1: 이미 계산된 예측값으로 평가
    print("="*70)
    print("EXAMPLE 1: Evaluation with predicted labels")
    print("="*70)
    
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 200)
    y_pred = np.random.randint(0, 3, 200)
    class_names = ['Northern', 'Midlands', 'Southern']
    
    evaluator = ModelEvaluator(y_true, y_pred, class_names)
    evaluator.full_evaluation(
        save_dir='./results/example1',
        show_plots=False,
        save_plots=True
    )
    
    # 예제 2: 간편 함수 사용
    print("\n" + "="*70)
    print("EXAMPLE 2: Using convenience function")
    print("="*70)
    
    evaluate_model(
        y_true=y_true, 
        y_pred=y_pred, 
        class_names=class_names,
        save_dir='./results/example2',
        show_plots=False
    )
    
    print("\n✅ All examples completed!")