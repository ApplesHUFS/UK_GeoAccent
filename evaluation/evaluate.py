"""
evaluation/evaluate.py
모델 평가 스크립트
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from models import GeoAccentClassifier
from data import EnglishDialectsDataset, collate_fn
from evaluation.metrics import AccentMetrics
from utils.config import REGION_LABELS, GENDER_LABELS, ID_TO_REGION, ID_TO_GENDER, REGION_COORDS


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(
        self,
        model,
        test_loader,
        device='cuda',
        output_dir='./results'
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics
        self.metrics = AccentMetrics()
        
    def evaluate(self):
        """모델 평가 실행"""
        
        print("\n" + "=" * 50)
        print("Model Evaluation")
        print("=" * 50)
        
        self.model.eval()
        
        # 결과 저장용
        all_region_preds = []
        all_region_labels = []
        all_gender_preds = []
        all_gender_labels = []
        all_coords_pred = []
        all_coords_true = []
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # 데이터 이동
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                coords = batch['coords'].to(self.device)
                region_labels = batch['region_labels'].to(self.device)
                gender_labels = batch['gender_labels'].to(self.device)
                
                # Forward pass
                # Prevent label leakage during evaluation: do not give true
                # coordinates to the model when computing predictions. Use
                # a zeroed coordinate input and compute true_geo_embedding
                # separately for metrics that require it.
                coords_zero = torch.zeros_like(coords)
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    coordinates=coords_zero
                )

                # compute true geo embedding for metric computations
                try:
                    outputs['true_geo_embedding'] = self.model.geo_embedding(coords)
                except Exception:
                    # if not available, rely on whatever the model returned
                    pass
                
                # Predictions
                region_preds = torch.argmax(outputs['region_logits'], dim=1)
                gender_preds = torch.argmax(outputs['gender_logits'], dim=1)
                
                # 결과 수집
                all_region_preds.extend(region_preds.cpu().numpy())
                all_region_labels.extend(region_labels.cpu().numpy())
                all_gender_preds.extend(gender_preds.cpu().numpy())
                all_gender_labels.extend(gender_labels.cpu().numpy())
                all_coords_pred.extend(outputs['predicted_geo_embedding'].cpu().numpy())
                all_coords_true.extend(outputs['true_geo_embedding'].cpu().numpy())
                
                num_batches += 1
        
        # NumPy 배열로 변환
        all_region_preds = np.array(all_region_preds)
        all_region_labels = np.array(all_region_labels)
        all_gender_preds = np.array(all_gender_preds)
        all_gender_labels = np.array(all_gender_labels)
        all_coords_pred = np.array(all_coords_pred)
        all_coords_true = np.array(all_coords_true)
        
        # Metrics 계산
        results = self.compute_metrics(
            all_region_preds,
            all_region_labels,
            all_gender_preds,
            all_gender_labels,
            all_coords_pred,
            all_coords_true
        )
        
        # 결과 출력
        self.print_results(results)
        
        # 결과 저장
        self.save_results(results, all_region_preds, all_region_labels)
        
        # Confusion matrix 시각화
        self.plot_confusion_matrix(all_region_labels, all_region_preds)
        
        return results
    
    def compute_metrics(
        self,
        region_preds,
        region_labels,
        gender_preds,
        gender_labels,
        coords_pred,
        coords_true
    ):
        """메트릭 계산"""
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Region metrics
        region_acc = accuracy_score(region_labels, region_preds)
        region_f1_macro = f1_score(region_labels, region_preds, average='macro')
        region_f1_weighted = f1_score(region_labels, region_preds, average='weighted')
        region_precision = precision_score(region_labels, region_preds, average='macro')
        region_recall = recall_score(region_labels, region_preds, average='macro')
        
        # Gender metrics
        gender_acc = accuracy_score(gender_labels, gender_preds)
        gender_f1 = f1_score(gender_labels, gender_preds, average='binary')
        
        # Per-class accuracy (Region)
        per_class_acc = {}
        for region_id, region_name in ID_TO_REGION.items():
            mask = region_labels == region_id
            if mask.sum() > 0:
                acc = accuracy_score(
                    region_labels[mask],
                    region_preds[mask]
                )
                per_class_acc[region_name] = acc
        
        # Geographic distance (optional - if you want to measure embedding quality)
        # 코사인 유사도
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = []
        for pred, true in zip(coords_pred, coords_true):
            sim = cosine_similarity([pred], [true])[0][0]
            cos_sim.append(sim)
        avg_cos_sim = np.mean(cos_sim)
        
        return {
            'region_accuracy': region_acc,
            'region_f1_macro': region_f1_macro,
            'region_f1_weighted': region_f1_weighted,
            'region_precision': region_precision,
            'region_recall': region_recall,
            'gender_accuracy': gender_acc,
            'gender_f1': gender_f1,
            'per_class_accuracy': per_class_acc,
            'avg_cosine_similarity': avg_cos_sim
        }
    
    def print_results(self, results):
        """결과 출력"""
        
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        
        print("\n[Region Classification]")
        print(f"  Accuracy:       {results['region_accuracy']:.4f}")
        print(f"  F1 (Macro):     {results['region_f1_macro']:.4f}")
        print(f"  F1 (Weighted):  {results['region_f1_weighted']:.4f}")
        print(f"  Precision:      {results['region_precision']:.4f}")
        print(f"  Recall:         {results['region_recall']:.4f}")
        
        print("\n[Gender Classification]")
        print(f"  Accuracy:       {results['gender_accuracy']:.4f}")
        print(f"  F1 Score:       {results['gender_f1']:.4f}")
        
        print("\n[Per-Class Accuracy (Region)]")
        for region_name, acc in results['per_class_accuracy'].items():
            print(f"  {region_name:12s}: {acc:.4f}")
        
        print(f"\n[Geographic Embedding]")
        print(f"  Avg Cosine Similarity: {results['avg_cosine_similarity']:.4f}")
        
        print("\n" + "=" * 50)
    
    def save_results(self, results, region_preds, region_labels):
        """결과를 JSON 파일로 저장"""
        
        # Per-class accuracy를 직렬화 가능하게 변환
        per_class_acc_serializable = {
            k: float(v) for k, v in results['per_class_accuracy'].items()
        }
        
        results_dict = {
            'region_accuracy': float(results['region_accuracy']),
            'region_f1_macro': float(results['region_f1_macro']),
            'region_f1_weighted': float(results['region_f1_weighted']),
            'region_precision': float(results['region_precision']),
            'region_recall': float(results['region_recall']),
            'gender_accuracy': float(results['gender_accuracy']),
            'gender_f1': float(results['gender_f1']),
            'per_class_accuracy': per_class_acc_serializable,
            'avg_cosine_similarity': float(results['avg_cosine_similarity'])
        }
        
        output_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"\n✅ Results saved to: {output_path}")
    
    def plot_confusion_matrix(self, true_labels, pred_labels):
        """Confusion Matrix 시각화"""
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[ID_TO_REGION[i] for i in range(len(REGION_LABELS))],
            yticklabels=[ID_TO_REGION[i] for i in range(len(REGION_LABELS))]
        )
        plt.title('Region Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {output_path}")


def evaluate_model(args):
    """평가 실행 함수"""
    
    print("=" * 50)
    print("GeoAccent Model Evaluation")
    print("=" * 50)
    
    # 1. 데이터셋 로드
    print(f"\n1. Loading {args.split} dataset...")
    test_dataset = EnglishDialectsDataset(
        split=args.split,
        use_augment=False,
        data_dir='./data/english_dialects'
    )
    print(f"   Samples: {len(test_dataset)}")
    
    # 2. DataLoader 생성
    print("\n2. Creating DataLoader...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 3. 모델 로드
    print("\n3. Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GeoAccentClassifier(
        model_name='facebook/wav2vec2-large-xlsr-53',
        num_regions=len(REGION_LABELS),
        num_genders=len(GENDER_LABELS),
        hidden_dim=1024,
        geo_embedding_dim=256,
        fusion_dim=512,
        dropout=0.1,
        freeze_lower_layers=False,  # 평가 시에는 freeze 해제
        num_frozen_layers=0
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"   Model loaded from: {args.checkpoint}")
    print(f"   Device: {device}")
    
    # 4. Evaluator 생성 및 평가
    print("\n4. Creating evaluator...")
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    # 5. 평가 실행
    results = evaluator.evaluate()
    
    print("\n" + "=" * 50)
    print("Evaluation completed!")
    print("=" * 50)
    
    return results


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Geo-Accent Classifier')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='체크포인트 경로'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['validation', 'test'],
        help='평가할 데이터 split'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='배치 크기'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='결과 저장 디렉토리'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
