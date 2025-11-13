import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
import librosa
import librosa.display

def plot_confusion_matrix(predictions, labels, class_names, title="Confusion Matrix", save_path=None):
    # Confusion matrix 시각화
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_training_curves(train_losses, val_losses, val_accuracies, save_path=None):
    # 학습 곡선 시각화
    epochs = np.arange(1, len(train_losses)+1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    # Loss plot
    axs[0].plot(epochs, train_losses, label='Train Loss', color='b')
    axs[0].plot(epochs, val_losses, label='Val Loss', color='orange')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Loss Curve')
    
    # Val accuracy plot
    axs[1].plot(epochs, val_accuracies, label='Val Acc', color='green')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].set_title('Validation Accuracy')
    
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_per_class_metrics(metrics_dict, class_names, save_path=None):
    # 클래스별 메트릭 (F1, Precision, Recall) 바 플롯
    f1s = [metrics_dict[class_]['f1'] for class_ in class_names]
    pre = [metrics_dict[class_]['precision'] for class_ in class_names]
    rec = [metrics_dict[class_]['recall'] for class_ in class_names]
    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, f1s, width, label='F1', color='skyblue')
    ax.bar(x, pre, width, label='Precision', color='orange')
    ax.bar(x + width, rec, width, label='Recall', color='lightgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30)
    ax.legend()
    ax.set_title('Per-class Metrics')
    
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_waveform(audio_path, title="Waveform", save_path=None):
    # 오디오 파형 시각화
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
