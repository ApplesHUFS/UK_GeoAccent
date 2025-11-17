"""
시각화 유틸리티 함수
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import librosa
import librosa.display
from typing import Dict, List, Optional


def plot_training_curves(train_losses: List[float], 
                         val_losses: List[float], 
                         val_accuracies: List[float], 
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    학습 곡선 시각화 (학습 중 실시간 모니터링용)
    
    Args:
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        val_accuracies: 검증 정확도 리스트
        save_path: 저장 경로
    
    Returns:
        Figure object
    """
    epochs = np.arange(1, len(train_losses) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss plot
    axs[0].plot(epochs, train_losses, label='Train Loss', color='b', linewidth=2)
    axs[0].plot(epochs, val_losses, label='Val Loss', color='orange', linewidth=2)
    axs[0].set_xlabel('Epoch', fontweight='bold', fontsize=11)
    axs[0].set_ylabel('Loss', fontweight='bold', fontsize=11)
    axs[0].legend(fontsize=10)
    axs[0].set_title('Loss Curve', fontweight='bold', fontsize=12, pad=10)
    axs[0].grid(alpha=0.3, linestyle='--')
    
    # Validation accuracy plot
    axs[1].plot(epochs, val_accuracies, label='Val Accuracy', 
                color='green', linewidth=2, marker='o', markersize=4)
    axs[1].set_xlabel('Epoch', fontweight='bold', fontsize=11)
    axs[1].set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    axs[1].legend(fontsize=10)
    axs[1].set_title('Validation Accuracy', fontweight='bold', fontsize=12, pad=10)
    axs[1].grid(alpha=0.3, linestyle='--')
    axs[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to {save_path}")
    
    return fig


def plot_per_class_metrics(metrics_dict: Dict[str, Dict[str, float]], 
                           class_names: List[str], 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    클래스별 F1, Precision, Recall 바 차트
    
    Args:
        metrics_dict: {class_name: {'f1': ..., 'precision': ..., 'recall': ...}}
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
    
    Returns:
        Figure object
    """
    f1s = [metrics_dict[class_]['f1'] for class_ in class_names]
    precisions = [metrics_dict[class_]['precision'] for class_ in class_names]
    recalls = [metrics_dict[class_]['recall'] for class_ in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, f1s, width, label='F1 Score', 
                   color='skyblue', edgecolor='navy', alpha=0.8)
    bars2 = ax.bar(x, precisions, width, label='Precision', 
                   color='orange', edgecolor='darkorange', alpha=0.8)
    bars3 = ax.bar(x + width, recalls, width, label='Recall', 
                   color='lightgreen', edgecolor='darkgreen', alpha=0.8)
    
    # 값 표시
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax.set_title('Per-Class Metrics Comparison', fontweight='bold', 
                fontsize=13, pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics saved to {save_path}")
    
    return fig


def plot_waveform(audio_path: str, 
                 title: str = "Audio Waveform", 
                 save_path: Optional[str] = None,
                 sr: Optional[int] = None) -> plt.Figure:
    """
    오디오 파형 시각화
    
    Args:
        audio_path: 오디오 파일 경로
        title: 그래프 제목
        save_path: 저장 경로
        sr: 샘플링 레이트 (None이면 원본 사용)
    
    Returns:
        Figure object
    """
    y, loaded_sr = librosa.load(audio_path, sr=sr)
    
    fig = plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=loaded_sr, alpha=0.7)
    
    plt.title(title, fontweight='bold', fontsize=12, pad=10)
    plt.xlabel('Time (seconds)', fontweight='bold', fontsize=10)
    plt.ylabel('Amplitude', fontweight='bold', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Waveform saved to {save_path}")
    
    return fig


def plot_spectrogram(audio_path: str,
                    title: str = "Mel Spectrogram",
                    save_path: Optional[str] = None,
                    sr: int = 22050,
                    n_mels: int = 128) -> plt.Figure:
    """
    오디오 스펙트로그램 시각화 (추가 기능)
    
    Args:
        audio_path: 오디오 파일 경로
        title: 그래프 제목
        save_path: 저장 경로
        sr: 샘플링 레이트
        n_mels: Mel 필터 개수
    
    Returns:
        Figure object
    """
    y, _ = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig = plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontweight='bold', fontsize=12, pad=10)
    plt.xlabel('Time (seconds)', fontweight='bold', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Spectrogram saved to {save_path}")
    
    return fig