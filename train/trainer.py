"""
train/trainer.py
훈련 로직
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import time

from evaluation.evaluate import ModelEvaluator
from utils.config import REGION_LABELS

class AccentTrainer:
    """지역 억양 분류 모델 학습 클래스 - 최적화 버전"""
    
    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        region_coords,
        device='cuda',
        learning_rate=1e-5,
        num_epochs=25,
        gradient_accumulation_steps=4,
        use_amp=True,
        amp_dtype='bfloat16',
        max_grad_norm=1.0,
        warmup_steps=500,
        early_stopping_patience=5,
        min_delta=0.001,
        save_steps=500,
        eval_steps=500,
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        use_wandb=False
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
            gradient_accumulation_steps: 그래디언트 누적 스텝
            use_amp: Mixed Precision 사용 여부
            amp_dtype: 'float16' or 'bfloat16'
            max_grad_norm: Gradient clipping norm
            warmup_steps: Learning rate warmup 스텝
            early_stopping_patience: Early stopping patience
            min_delta: Early stopping 최소 개선 폭
            save_steps: 체크포인트 저장 주기
            eval_steps: 검증 주기
            checkpoint_dir: 체크포인트 저장 경로
            log_dir: 로그 저장 경로
            use_wandb: Weights & Biases 사용 여부
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.region_coords = region_coords
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        # Mixed Precision Training
        self.use_amp = use_amp and device == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            self.amp_dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
            print(f"✅ Using AMP with {amp_dtype}")
        
        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler: Warmup + Cosine Annealing
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # Warmup Scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Main Scheduler
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )
        
        # Early Stopping
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.should_stop = False
        
        # 디렉토리
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 최고 성능 추적
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        
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
            'val_gender_acc': [],
            'learning_rates': []
        }
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                print("✅ Weights & Biases enabled")
            except ImportError:
                print("⚠️  wandb not installed, disabling W&B logging")
                self.use_wandb = False
        
        # 시간 측정
        self.epoch_start_time = None
        self.total_train_time = 0
    
    def _get_coordinates_tensor(self, region_names):
        """지역 이름 리스트 -> 좌표 텐서 변환"""
        coords = []
        for region in region_names:
            region_key = region.lower()
            if region_key in self.region_coords:
                coords.append(self.region_coords[region_key])
            else:
                raise ValueError(f"Unknown region: {region}")
        return torch.FloatTensor(coords).to(self.device)
    
    def _update_scheduler(self):
        """Learning rate scheduler 업데이트"""
        if self.global_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
    
    def _log_metrics(self, metrics, prefix='train'):
        """메트릭 로깅 (W&B 지원)"""
        if self.use_wandb:
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
            log_dict['global_step'] = self.global_step
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            self.wandb.log(log_dict)
    
    def train_epoch(self, epoch):
        """
        한 에포크 학습
        
        Returns:
            에포크별 메트릭 딕셔너리
        """
        self.model.train()
        self.epoch_start_time = time.time()
        
        total_loss_sum = 0
        region_loss_sum = 0
        gender_loss_sum = 0
        distance_loss_sum = 0
        
        region_preds, region_labels_list = [], []
        gender_preds, gender_labels_list = [], []
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        for step, batch in enumerate(pbar):
            # 배치 언팩
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            region_labels = batch['region_labels'].to(self.device)
            gender_labels = batch['gender_labels'].to(self.device)
            coordinates = batch['coords'].to(self.device)
            
            # Mixed Precision Forward
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        input_values,
                        attention_mask=attention_mask,
                        coordinates=coordinates
                    )
                    
                    total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                        outputs, region_labels, gender_labels
                    )
                    # Gradient Accumulation
                    total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward (AMP)
                self.scaler.scale(total_loss).backward()
            else:
                # Standard Forward & Backward
                outputs = self.model(
                    input_values,
                    attention_mask=attention_mask,
                    coordinates=coordinates
                )
                
                total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                    outputs, region_labels, gender_labels
                )
                total_loss = total_loss / self.gradient_accumulation_steps
                total_loss.backward()
            
            # Optimizer Step (Gradient Accumulation 고려)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self._update_scheduler()
                self.global_step += 1
                
                # 주기적 검증
                if self.global_step % self.eval_steps == 0:
                    val_metrics = self.validate()
                    self._log_metrics(val_metrics, prefix='val')
                    self.model.train()
                
                # 주기적 체크포인트 저장
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(
                        epoch=epoch,
                        val_acc=self.best_val_acc,
                        is_best=False,
                        step=self.global_step
                    )
            
            # 메트릭 누적 (원래 스케일로 복원)
            total_loss_sum += (total_loss.item() * self.gradient_accumulation_steps)
            region_loss_sum += region_loss.item()
            gender_loss_sum += gender_loss.item()
            distance_loss_sum += distance_loss.item()
            
            region_preds.extend(outputs['region_logits'].argmax(dim=-1).cpu().numpy())
            region_labels_list.extend(region_labels.cpu().numpy())
            gender_preds.extend(outputs['gender_logits'].argmax(dim=-1).cpu().numpy())
            gender_labels_list.extend(gender_labels.cpu().numpy())
            
            # Progress bar 업데이트
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{total_loss.item() * self.gradient_accumulation_steps:.4f}',
                'r_loss': f'{region_loss.item():.4f}',
                'd_loss': f'{distance_loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })
        
        # 에포크 메트릭 계산
        num_batches = len(self.train_loader)
        avg_total_loss = total_loss_sum / num_batches
        avg_region_loss = region_loss_sum / num_batches
        avg_gender_loss = gender_loss_sum / num_batches
        avg_distance_loss = distance_loss_sum / num_batches
        
        region_acc = accuracy_score(region_labels_list, region_preds)
        gender_acc = accuracy_score(gender_labels_list, gender_preds)
        
        epoch_time = time.time() - self.epoch_start_time
        self.total_train_time += epoch_time
        
        return {
            'total_loss': avg_total_loss,
            'region_loss': avg_region_loss,
            'gender_loss': avg_gender_loss,
            'distance_loss': avg_distance_loss,
            'region_acc': region_acc,
            'gender_acc': gender_acc,
            'epoch_time': epoch_time,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, epoch=None, save_confusion_matrix=False):
        """검증"""
        self.model.eval()
        
        total_loss_sum = 0
        region_preds, region_labels_list = [], []
        gender_preds, gender_labels_list = [], []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating', leave=False):
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                region_labels = batch['region_labels'].to(self.device)
                gender_labels = batch['gender_labels'].to(self.device)
                coordinates = batch['coords'].to(self.device)
                
                # Forward (Mixed Precision)
                if self.use_amp:
                    with autocast(dtype=self.amp_dtype):
                        outputs = self.model(
                            input_values,
                            attention_mask=attention_mask,
                            coordinates=coordinates
                        )
                        total_loss, _, _, _ = self.criterion(
                            outputs, region_labels, gender_labels
                        )
                else:
                    outputs = self.model(
                        input_values,
                        attention_mask=attention_mask,
                        coordinates=coordinates
                    )
                    total_loss, _, _, _ = self.criterion(
                        outputs, region_labels, gender_labels
                    )
                
                total_loss_sum += total_loss.item()
                region_preds.extend(outputs['region_logits'].argmax(dim=-1).cpu().numpy())
                region_labels_list.extend(region_labels.cpu().numpy())
                gender_preds.extend(outputs['gender_logits'].argmax(dim=-1).cpu().numpy())
                gender_labels_list.extend(gender_labels.cpu().numpy())
                
                if outputs['attention_weights'] is not None:
                    attention_weights_list.append(outputs['attention_weights'].cpu().numpy())
        
        # 메트릭 계산
        avg_loss = total_loss_sum / len(self.val_loader)
        region_acc = accuracy_score(region_labels_list, region_preds)
        region_f1 = f1_score(region_labels_list, region_preds, average='weighted')
        gender_acc = accuracy_score(gender_labels_list, gender_preds)
        
        # Confusion Matrix 저장
        if save_confusion_matrix and epoch is not None:
            evaluator = ModelEvaluator(
                y_true=np.array(region_labels_list),
                y_pred=np.array(region_preds),
                class_names=list(REGION_LABELS.keys())
            )
            
            cm_path = os.path.join(self.log_dir, f'confusion_matrix_epoch_{epoch}.png')
            evaluator.plot_confusion_matrix(save_path=cm_path, show_percentages=True)
            print(f"  📊 Confusion matrix saved to {cm_path}")
        
        return {
            'loss': avg_loss,
            'region_acc': region_acc,
            'region_f1': region_f1,
            'gender_acc': gender_acc,
            'region_preds': region_preds,
            'region_labels': region_labels_list,
            'attention_weights': np.concatenate(attention_weights_list) if attention_weights_list else None
        }
    
    def check_early_stopping(self, val_acc):
        """Early Stopping 체크"""
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            return True  # Improved
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.should_stop = True
                print(f"\n⚠️  Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
            return False
    
    def save_checkpoint(self, epoch, val_acc, is_best=False, step=None):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'main_scheduler_state_dict': self.main_scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Latest checkpoint
        filename = f'checkpoint_step_{step}.pt' if step else 'latest_checkpoint.pt'
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved! Val Acc: {val_acc:.4f}")
    
    def plot_history(self):
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Total Loss
        axes[0, 0].plot(self.history['train_total_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_total_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Component Losses
        axes[0, 1].plot(self.history['train_region_loss'], label='Region', linewidth=2)
        axes[0, 1].plot(self.history['train_gender_loss'], label='Gender', linewidth=2)
        axes[0, 1].plot(self.history['train_distance_loss'], label='Distance', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Learning Rate
        axes[0, 2].plot(self.history['learning_rates'], linewidth=2, color='green')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Region Accuracy
        axes[1, 0].plot(self.history['train_region_acc'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history['val_region_acc'], label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Region Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Gender Accuracy
        axes[1, 1].plot(self.history['train_gender_acc'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history['val_gender_acc'], label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Gender Classification Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Training Time
        if hasattr(self, 'epoch_times'):
            axes[1, 2].plot(self.epoch_times, linewidth=2, color='purple')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].set_title('Epoch Training Time')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 Training history saved to {save_path}")
    
    def train(self):
        """전체 학습 프로세스"""
        print("\n" + "="*70)
        print("Starting Geo-Accent Classifier Training")
        print(f"Total steps: {self.total_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print("="*70)
        
        self.epoch_times = []
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print('='*70)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.epoch_times.append(train_metrics['epoch_time'])
            
            # Validate
            save_cm = (epoch % 5 == 0)
            val_metrics = self.validate(epoch=epoch, save_confusion_matrix=save_cm)
            
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
            self.history['learning_rates'].append(train_metrics['lr'])
            
            # 로깅
            self._log_metrics(train_metrics, prefix='train')
            self._log_metrics(val_metrics, prefix='val')
            
            # 결과 출력
            print(f"\n📊 Training Metrics:")
            print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Region Acc: {train_metrics['region_acc']:.4f}")
            print(f"  Gender Acc: {train_metrics['gender_acc']:.4f}")
            print(f"  Epoch Time: {train_metrics['epoch_time']:.2f}s")
            
            print(f"\n📊 Validation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Region Acc: {val_metrics['region_acc']:.4f}")
            print(f"  Region F1: {val_metrics['region_f1']:.4f}")
            print(f"  Gender Acc: {val_metrics['gender_acc']:.4f}")
            
            # Early Stopping & Checkpoint
            improved = self.check_early_stopping(val_metrics['region_acc'])
            if improved:
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics['region_acc'], is_best=True)
            
            if self.should_stop:
                print(f"\n🛑 Training stopped at epoch {epoch}")
                break
            
            # 주기적 히스토리 시각화
            if epoch % 5 == 0:
                self.plot_history()
        
        # 학습 완료
        print("\n" + "="*70)
        print("Training Completed!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f} at Epoch {self.best_epoch}")
        print(f"Total Training Time: {self.total_train_time / 3600:.2f} hours")
        print(f"Average Epoch Time: {np.mean(self.epoch_times):.2f} seconds")
        print("="*70 + "\n")
        
        self.plot_history()
        self.validate(epoch='final', save_confusion_matrix=True)