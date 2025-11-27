"""
evaluation/evaluate.py
Script to evaluate the GeoAccentClassifier model on the test set.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, mean_squared_error, mean_absolute_error, 
    r2_score
)
from sklearn.metrics.pairwise import cosine_similarity

from data.dataset import EnglishDialectsDataset, collate_fn
from models.classifier import GeoAccentClassifier
from utils.config import REGION_LABELS, GENDER_LABELS


class ModelEvaluator:
    def __init__(self, args):
        self.args = args

        if args.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif args.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available. Switching to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            
        print(f"Evaluation will run on {self.device}")

        # 1. Dataset and DataLoader setup
        print("1. Loading test dataset...")
        self.test_dataset = EnglishDialectsDataset(
            split='test', 
            audio_sample_rate=16000, 
            data_dir="./data/english_dialects"
        )
        
        # DataLoader 생성 
        print("\n2. Creating DataLoader...")
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,     
            collate_fn=collate_fn,
            pin_memory=True  
        )

        # 2. Model setup
        print("\n3. Loading model and checkpoint...")
        self.model = GeoAccentClassifier(
            num_regions=len(REGION_LABELS),
            num_genders=len(GENDER_LABELS)
        ).to(self.device)
        
        # Checkpoint loading
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

        self.model.eval()
        
    def evaluate(self):
        all_region_predictions = []
        all_region_labels = []
        all_gender_predictions = []
        all_gender_labels = []
        
        all_coords_true = []
        all_coords_pred = []

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                
                input_values = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                coordinates = batch["coords"].to(self.device)
                
                # ✅ use_fusion 여부에 따라 coordinates 전달
                input_coords = coordinates if self.model.use_fusion else None
                
                # Forward pass
                model_output = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    coordinates=input_coords
                )

                region_logits = model_output['region_logits']
                gender_logits = model_output['gender_logits']
                coords_pred = model_output['predicted_geo_embedding']
                
                # --- Region Classification ---
                region_pred = torch.argmax(region_logits, dim=-1)
                
                all_region_predictions.extend(region_pred.cpu().tolist())
                all_region_labels.extend(batch["region_labels"].cpu().tolist())
                
                # --- Gender Classification ---
                gender_pred = torch.argmax(gender_logits, dim=-1)
                
                all_gender_predictions.extend(gender_pred.cpu().tolist())
                all_gender_labels.extend(batch["gender_labels"].cpu().tolist())
                
                # --- Coordinate Regression (use_fusion=True일 때만) ---
                if coords_pred is not None:
                    all_coords_pred.append(coords_pred.cpu().numpy())
                    all_coords_true.append(batch["coords"].cpu().numpy())

        # -----------------------------------------------------------
        # 4. Results Aggregation and Metrics Calculation
        # -----------------------------------------------------------
        
        # ✅ use_fusion=False면 coordinate metrics 계산 생략
        if len(all_coords_pred) > 0:
            coords_true = np.concatenate(all_coords_true, axis=0)
            coords_pred = np.concatenate(all_coords_pred, axis=0)
            avg_cosine_similarity = np.mean(
                cosine_similarity(coords_true, coords_pred)
            )
        else:
            avg_cosine_similarity = None

        # --- REGION METRICS ---
        region_acc = accuracy_score(all_region_labels, all_region_predictions)
        region_f1_macro = f1_score(all_region_labels, all_region_predictions, average='macro')
        region_f1_weighted = f1_score(all_region_labels, all_region_predictions, average='weighted')
        region_precision = precision_score(all_region_labels, all_region_predictions, average='weighted')
        region_recall = recall_score(all_region_labels, all_region_predictions, average='weighted')
        
        # --- GENDER METRICS ---
        gender_acc = accuracy_score(all_gender_labels, all_gender_predictions)
        gender_f1 = f1_score(all_gender_labels, all_gender_predictions, average='binary', pos_label=1)

        # --- PER-CLASS ACCURACY ---
        per_class_accuracy = {}
        unique_labels = np.unique(all_region_labels)
        label_to_name = {v: k for k, v in REGION_LABELS.items()} 
        
        for label_id in unique_labels:
            class_indices = np.where(np.array(all_region_labels) == label_id)
            class_predictions = np.array(all_region_predictions)[class_indices]
            class_labels = np.array(all_region_labels)[class_indices]
            
            acc = accuracy_score(class_labels, class_predictions)
            per_class_accuracy[label_to_name[label_id]] = acc


        results = {
            "region_accuracy": region_acc,
            "region_f1_macro": region_f1_macro,
            "region_f1_weighted": region_f1_weighted,
            "region_precision": region_precision,
            "region_recall": region_recall,
            "gender_accuracy": gender_acc,
            "gender_f1": gender_f1,
            "per_class_accuracy": per_class_accuracy,
            "avg_cosine_similarity": avg_cosine_similarity
        }
        
        class_names = list(REGION_LABELS.keys()) 
        labels_indices = np.arange(len(class_names))

        print("\n\n" + "="*50)
        print("--- Region Confusion Matrix Analysis ---")
        print("="*50)

        # 1. Raw Confusion Matrix 
        conf_matrix = confusion_matrix(
            y_true=all_region_labels, 
            y_pred=all_region_predictions, 
            labels=labels_indices
        )

        print("\n[1] Raw Confusion Matrix (Count):")
        conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
        print(conf_df)

        # 2. Normalized Confusion Matrix (정확도 비율)
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        print("\n[2] Normalized Confusion Matrix (Accuracy % by True Label):")
        conf_df_norm = pd.DataFrame(conf_matrix_norm, index=class_names, columns=class_names)
        print(conf_df_norm.round(4)) 

        return results


def evaluate_model(args):
    """
    Main function to run the evaluation process.
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    evaluator = ModelEvaluator(args)
    results = evaluator.evaluate()

    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*50)
    print("✅ Evaluation Completed")
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(results, indent=4))
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint (.pt)")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'], help="Dataset split to evaluate")
    parser.add_argument('--output_dir', type=str, default='./results/evaluation', help="Directory to save output metrics")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Evaluation device (cuda/cpu)')

    args = parser.parse_args()
    evaluate_model(args)