import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

from models.accent_classifier import AccentClassifierWithGeo, MultiTaskLossWithDistance
# TODO: 데이터로더 구현 후 import
# from data.dataset import EnglishDialectsDataset


# 지역 좌표 (normalized to [-1, 1])
REGION_COORDS = {
    'irish': (0.533, -0.626),      # (53.3, -62.6) / 100
    'midlands': (0.526, -0.114),   # Birmingham
    'northern': (0.546, -0.593),   # Belfast
    'scottish': (0.559, -0.319),   # Edinburgh
    'southern': (0.515, -0.013),   # London
    'welsh': (0.514, -0.318)       # Cardiff
}

REGION_TO_IDX = {
    'irish': 0,
    'midlands': 1,
    'northern': 2,
    'scottish': 3,
    'southern': 4,
    'welsh': 5
}


class NovelTrainer:
    """Novel 모델 학습 클래스"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-5,
        num_epochs=30,
        checkpoint_dir='./checkpoints_novel',
        log_dir='./logs_novel'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Optimizer: 낮은 learning rate (partial fine-tuning)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Loss function
        self.criterion = MultiTaskLossWithDistance(
            region_weight=1.0,
            gender_weight=0.3,
            distance_weight=0.5,
            distance_metric='cosine'
        )
        
        # Directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # History
        self.history = {
            'train_total_loss': [],
            'train_region_loss': [],
            'train_gender_loss': [],
            'train_distance_loss': [],
            'val_total_loss': [],
            'train_region_acc': [],
            'val_region_acc': [],
            'train_gender_acc': [],
            'val_gender_acc': []
        }
    
    def _get_coordinates_tensor(self, region_names):
        """지역 이름 → 좌표 텐서 변환"""
        coords = []
        for region in region_names:
            region_key = region.lower()
            coords.append(REGION_COORDS[region_key])
        return torch.FloatTensor(coords).to(self.device)
    
    def train_epoch(self):
        """한 에폭 학습"""
        self.model.train()
        
        total_loss_sum = 0
        region_loss_sum = 0
        gender_loss_sum = 0
        distance_loss_sum = 0
        
        region_preds, region_labels_list = [], []
        gender_preds, gender_labels_list = [], []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # TODO: 실제 데이터로더 구현 후 수정
            # batch 구조:
            # {
            #   'input_values': (B, seq_len),
            #   'attention_mask': (B, seq_len),
            #   'region': (B,) - region label indices,
            #   'region_name': list of region names (for coordinate lookup),
            #   'gender': (B,) - gender label indices
            # }
            
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            region_labels = batch['region'].to(self.device)
            gender_labels = batch['gender'].to(self.device)
            
            # 지역 좌표 가져오기
            coordinates = self._get_coordinates_tensor(batch['region_name'])
            
            # Forward
            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
                coordinates=coordinates
            )
            
            # Loss
            total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                outputs, region_labels, gender_labels
            )
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss_sum += total_loss.item()
            region_loss_sum += region_loss.item()
            gender_loss_sum += gender_loss.item()
            distance_loss_sum += distance_loss.item()
            
            region_preds.extend(outputs['region_logits'].argmax(dim=-1).cpu().numpy())
            region_labels_list.extend(region_labels.cpu().numpy())
            gender_preds.extend(outputs['gender_logits'].argmax(dim=-1).cpu().numpy())
            gender_labels_list.extend(gender_labels.cpu().numpy())
            
            # Progress bar
            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'region_loss': f'{region_loss.item():.4f}',
                'dist_loss': f'{distance_loss.item():.4f}'
            })
        
        # Epoch metrics
        num_batches = len(self.train_loader)
        avg_total_loss = total_loss_sum / num_batches
        avg_region_loss = region_loss_sum / num_batches
        avg_gender_loss = gender_loss_sum / num_batches
        avg_distance_loss = distance_loss_sum / num_batches
        
        region_acc = accuracy_score(region_labels_list, region_preds)
        gender_acc = accuracy_score(gender_labels_list, gender_preds)
        
        return {
            'total_loss': avg_total_loss,
            'region_loss': avg_region_loss,
            'gender_loss': avg_gender_loss,
            'distance_loss': avg_distance_loss,
            'region_acc': region_acc,
            'gender_acc': gender_acc
        }
    
    def validate(self):
        """검증"""
        self.model.eval()
        
        total_loss_sum = 0
        region_preds, region_labels_list = [], []
        gender_preds, gender_labels_list = [], []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                region_labels = batch['region'].to(self.device)
                gender_labels = batch['gender'].to(self.device)
                
                # 지역 좌표
                coordinates = self._get_coordinates_tensor(batch['region_name'])
                
                # Forward
                outputs = self.model(
                    input_values,
                    attention_mask=attention_mask,
                    coordinates=coordinates
                )
                
                # Loss
                total_loss, _, _, _ = self.criterion(
                    outputs, region_labels, gender_labels
                )
                
                total_loss_sum += total_loss.item()
                region_preds.extend(outputs['region_logits'].argmax(dim=-1).cpu().numpy())
                region_labels_list.extend(region_labels.cpu().numpy())
                gender_preds.extend(outputs['gender_logits'].argmax(dim=-1).cpu().numpy())
                gender_labels_list.extend(gender_labels.cpu().numpy())
                
                # Attention weights 저장 (시각화용)
                if outputs['attention_weights'] is not None:
                    attention_weights_list.append(outputs['attention_weights'].cpu().numpy())
        
        # Metrics
        avg_loss = total_loss_sum / len(self.val_loader)
        region_acc = accuracy_score(region_labels_list, region_preds)
        region_f1 = f1_score(region_labels_list, region_preds, average='weighted')
        gender_acc = accuracy_score(gender_labels_list, gender_preds)
        
        return {
            'loss': avg_loss,
            'region_acc': region_acc,
            'region_f1': region_f1,
            'gender_acc': gender_acc,
            'region_preds': region_preds,
            'region_labels': region_labels_list,
            'attention_weights': np.concatenate(attention_weights_list) if attention_weights_list else None
        }
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Latest checkpoint
        path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, path)
        
        # Best checkpoint
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, path)
            print(f"💾 Best model saved! Val Acc: {val_acc:.4f}")
    
    def plot_history(self):
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Total Loss
        axes[0, 0].plot(self.history['train_total_loss'], label='Train Total Loss')
        axes[0, 0].plot(self.history['val_total_loss'], label='Val Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Component Losses
        axes[0, 1].plot(self.history['train_region_loss'], label='Region Loss')
        axes[0, 1].plot(self.history['train_gender_loss'], label='Gender Loss')
        axes[0, 1].plot(self.history['train_distance_loss'], label='Distance Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Component Losses (Training)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Region Accuracy
        axes[1, 0].plot(self.history['train_region_acc'], label='Train Region Acc')
        axes[1, 0].plot(self.history['val_region_acc'], label='Val Region Acc')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Region Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Gender Accuracy
        axes[1, 1].plot(self.history['train_gender_acc'], label='Train Gender Acc')
        axes[1, 1].plot(self.history['val_gender_acc'], label='Val Gender Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Gender Classification Accuracy (Aux Task)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 Training history saved to {save_path}")
    
    def train(self):
        """전체 학습 프로세스"""
        print("\n" + "="*70)
        print("Starting Novel Model Training")
        print("="*70)
        self.model.print_model_info()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print('='*70)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # History
            self.history['train_total_loss'].append(train_metrics['total_loss'])
            self.history['train_region_loss'].append(train_metrics['region_loss'])
            self.history['train_gender_loss'].append(train_metrics['gender_loss'])
            self.history['train_distance_loss'].append(train_metrics['distance_loss'])
            self.history['val_total_loss'].append(val_metrics['loss'])
            self.history['train_region_acc'].append(train_metrics['region_acc'])
            self.history['val_region_acc'].append(val_metrics['region_acc'])
            self.history['train_gender_acc'].append(train_metrics['gender_acc'])
            self.history['val_gender_acc'].append(val_metrics['gender_acc'])
            
            # Print results
            print(f"\n📊 Training Metrics:")
            print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Region Loss: {train_metrics['region_loss']:.4f}")
            print(f"  Gender Loss: {train_metrics['gender_loss']:.4f}")
            print(f"  Distance Loss: {train_metrics['distance_loss']:.4f}")
            print(f"  Region Acc: {train_metrics['region_acc']:.4f}")
            print(f"  Gender Acc: {train_metrics['gender_acc']:.4f}")
            
            print(f"\n📊 Validation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Region Acc: {val_metrics['region_acc']:.4f}")
            print(f"  Region F1: {val_metrics['region_f1']:.4f}")
            print(f"  Gender Acc: {val_metrics['gender_acc']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = val_metrics['region_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['region_acc']
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics['region_acc'], is_best)
            
            # Plot every 5 epochs
            if epoch % 5 == 0:
                self.plot_history()
        
        print("\n" + "="*70)
        print("Training Completed!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f} at Epoch {self.best_epoch}")
        print("="*70 + "\n")
        
        # Final plot
        self.plot_history()


def main():
    """메인 학습 함수"""
    
    # Configuration
    config = {
        'model_name': 'facebook/wav2vec2-large-xlsr-53',
        'batch_size': 8,  # Large model이므로 작은 배치
        'learning_rate': 1e-5,  # Partial fine-tuning이므로 낮은 LR
        'num_epochs': 30,
        'num_frozen_layers': 8,  # 하위 8개 레이어 freeze
        'geo_embedding_dim': 256,
        'fusion_dim': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # TODO: 실제 데이터로더 구현 후 교체
    # train_dataset = EnglishDialectsDataset(split='train', augment=True)
    # val_dataset = EnglishDialectsDataset(split='val', augment=False)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    print("\n⚠️  Using dummy data loaders for testing")
    print("TODO: Replace with actual EnglishDialectsDataset")
    
    # Model
    model = AccentClassifierWithGeo(
        model_name=config['model_name'],
        num_regions=6,
        num_genders=2,
        geo_embedding_dim=config['geo_embedding_dim'],
        fusion_dim=config['fusion_dim'],
        freeze_lower_layers=True,
        num_frozen_layers=config['num_frozen_layers']
    )
    
    # Trainer
    # trainer = NovelTrainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=config['device'],
    #     learning_rate=config['learning_rate'],
    #     num_epochs=config['num_epochs']
    # )
    
    # trainer.train()


if __name__ == "__main__":
    main()