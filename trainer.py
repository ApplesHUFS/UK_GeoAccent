"""
trainer.py
모델 학습 및 검증 로직
"""

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os


class GeoAccentTrainer:
    """지역 억양 분류 모델 학습 클래스"""
    
    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        region_coords,
        device='cuda',
        learning_rate=1e-5,
        num_epochs=30,
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    ):
        """
        Args:
            model: GeoAccentClassifier 인스턴스
            criterion: MultiTaskLossWithDistance 인스턴스
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            region_coords: 지역명 -> (lat, lon) 좌표 딕셔너리
            device: 'cuda' 또는 'cpu'
            learning_rate: 학습률
            num_epochs: 에포크 수
            checkpoint_dir: 체크포인트 저장 경로
            log_dir: 로그 저장 경로
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.region_coords = region_coords
        self.device = device
        self.num_epochs = num_epochs
        
        # Optimizer: AdamW (partial fine-tuning용)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler: Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # 디렉토리
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 최고 성능 추적
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # 학습 히스토리
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
    
    def _get_coordinates_tensor(self, region_names): #pre-processing에서 처리하는게 좋아보임
        """
        지역 이름 리스트 -> 좌표 텐서 변환
        
        Args:
            region_names: 지역명 리스트
        
        Returns:
            좌표 텐서 (B, 2)
        """
        coords = []
        for region in region_names:
            region_key = region.lower()
            if region_key in self.region_coords:
                coords.append(self.region_coords[region_key])
            else:
                raise ValueError(f"Unknown region: {region}")
        return torch.FloatTensor(coords).to(self.device)
    
    def train_epoch(self):
        """
        한 에포크 학습
        
        Returns:
            에포크별 메트릭 딕셔너리
        """
        self.model.train()
        
        total_loss_sum = 0
        region_loss_sum = 0
        gender_loss_sum = 0
        distance_loss_sum = 0
        
        region_preds, region_labels_list = [], []
        gender_preds, gender_labels_list = [], []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # 배치 언팩
            input_values = batch['input_values'].to(self.device)  # (B, seq_len)
            attention_mask = batch['attention_mask'].to(self.device)  # (B, seq_len)
            region_labels = batch['region_labels'].to(self.device)  # (B,) - 정수 인덱스
            gender_labels = batch['gender_labels'].to(self.device)  # (B,)
            
            # 지역 좌표 가져오기
            coordinates = self._get_coordinates_tensor(batch['region_name'])  # (B, 2)
            
            # Forward
            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
                coordinates=coordinates
            )
            
            # Loss 계산
            total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                outputs, region_labels, gender_labels
            )
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 메트릭 누적
            total_loss_sum += total_loss.item()
            region_loss_sum += region_loss.item()
            gender_loss_sum += gender_loss.item()
            distance_loss_sum += distance_loss.item()
            
            region_preds.extend(outputs['region_logits'].argmax(dim=-1).cpu().numpy())
            region_labels_list.extend(region_labels.cpu().numpy())
            gender_preds.extend(outputs['gender_logits'].argmax(dim=-1).cpu().numpy())
            gender_labels_list.extend(gender_labels.cpu().numpy())
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'region_loss': f'{region_loss.item():.4f}',
                'dist_loss': f'{distance_loss.item():.4f}'
            })
        
        # 에포크 메트릭 계산
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
        """
        검증
        
        Returns:
            검증 메트릭 딕셔너리
        """
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
                
                # Loss (region_loss만 필요)
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
        
        # 메트릭 계산
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
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 에포크
            val_acc: 검증 정확도
            is_best: 최고 성능 모델 여부
        """
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
        axes[0, 0].plot(self.history['train_total_loss'], label='Train Total Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_total_loss'], label='Val Total Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Component Losses
        axes[0, 1].plot(self.history['train_region_loss'], label='Region Loss', linewidth=2)
        axes[0, 1].plot(self.history['train_gender_loss'], label='Gender Loss', linewidth=2)
        axes[0, 1].plot(self.history['train_distance_loss'], label='Distance Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_title('Component Losses (Training)', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Region Accuracy
        axes[1, 0].plot(self.history['train_region_acc'], label='Train Region Acc', linewidth=2)
        axes[1, 0].plot(self.history['val_region_acc'], label='Val Region Acc', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy', fontsize=12)
        axes[1, 0].set_title('Region Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Gender Accuracy
        axes[1, 1].plot(self.history['train_gender_acc'], label='Train Gender Acc', linewidth=2)
        axes[1, 1].plot(self.history['val_gender_acc'], label='Val Gender Acc', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy', fontsize=12)
        axes[1, 1].set_title('Gender Classification Accuracy (Auxiliary Task)', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 Training history saved to {save_path}")
    
    def train(self):
        """전체 학습 프로세스"""
        print("\n" + "="*70)
        print("Starting Geo-Accent Classifier Training")
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
            
            # History 기록
            self.history['train_total_loss'].append(train_metrics['total_loss'])
            self.history['train_region_loss'].append(train_metrics['region_loss'])
            self.history['train_gender_loss'].append(train_metrics['gender_loss'])
            self.history['train_distance_loss'].append(train_metrics['distance_loss'])
            self.history['val_total_loss'].append(val_metrics['loss'])
            self.history['train_region_acc'].append(train_metrics['region_acc'])
            self.history['val_region_acc'].append(val_metrics['region_acc'])
            self.history['train_gender_acc'].append(train_metrics['gender_acc'])
            self.history['val_gender_acc'].append(val_metrics['gender_acc'])
            
            # 결과 출력
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
            
            # 체크포인트 저장
            is_best = val_metrics['region_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['region_acc']
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics['region_acc'], is_best)
            
            # 매 5 에포크마다 히스토리 시각화
            if epoch % 5 == 0:
                self.plot_history()
        
        print("\n" + "="*70)
        print("Training Completed!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f} at Epoch {self.best_epoch}")
        print("="*70 + "\n")
        
        # 최종 히스토리 시각화
        self.plot_history()
