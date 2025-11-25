import os
import glob
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class AccentTrainer:
    """
    GeoAccent Classifier 학습 트레이너
    """
    
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
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.region_coords = region_coords
        self.device = device
        
        # Training params
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # Logging & Checkpointing
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # AMP
        self.scaler = GradScaler() if use_amp else None
        self.amp_dtype = torch.float16 if use_amp else torch.float32
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_accuracy': self.best_accuracy,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'model_config': self.model.get_config()
        }
        
        temp_path = os.path.join(self.checkpoint_dir, 'temp_checkpoint.pt')
        
        try:
            torch.save(checkpoint, temp_path)
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best.pt')
                if os.path.exists(best_path):
                    os.remove(best_path)
                os.rename(temp_path, best_path)
                print(f'✅ Saved best checkpoint (epoch {epoch}, acc={metrics.get("val_accuracy", 0):.4f})')
            else:
                last_path = os.path.join(self.checkpoint_dir, 'last.pt')
                if os.path.exists(last_path):
                    os.remove(last_path)
                os.rename(temp_path, last_path)
                print(f'💾 Saved last checkpoint (epoch {epoch})')
            
            self._cleanup_old_checkpoints(keep_last_n=1)
                
        except RuntimeError as e:
            print(f"❌ Checkpoint save failed: {e}")
            print("💡 Checking disk space...")
            os.system('df -h | grep /root')
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print("🗑️ Cleaning old checkpoints...")
            self._cleanup_old_checkpoints(keep_last_n=0)
            
            try:
                torch.save(checkpoint, temp_path)
                last_path = os.path.join(self.checkpoint_dir, 'last.pt')
                os.rename(temp_path, last_path)
                print("✅ Retry successful")
            except Exception as e2:
                print(f"❌ Retry failed: {e2}")
                print("⚠️ Continuing without saving checkpoint...")
    
    def _cleanup_old_checkpoints(self, keep_last_n=1):
        """오래된 체크포인트 정리"""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')
        checkpoints = sorted(glob.glob(pattern))
        
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints:
                try:
                    os.remove(old_ckpt)
                    print(f'🗑️ Removed: {os.path.basename(old_ckpt)}')
                except Exception as e:
                    print(f'Failed to remove {old_ckpt}: {e}')
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Model weights loaded")
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("   ✅ Optimizer state loaded")
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("   ✅ Scheduler state loaded")
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("   ✅ AMP scaler loaded")
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✅ Resumed from epoch {self.start_epoch}, step {self.global_step}")
        print(f"   Best accuracy so far: {self.best_accuracy:.4f}")
        
        return checkpoint
    
    def train_epoch(self, epoch):
        """한 epoch 학습"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            with autocast('cuda', dtype=self.amp_dtype):
                # Prevent label leakage: do NOT pass true coords into the
                # model when computing logits. Provide a neutral (zero)
                # coordinate input instead so the model cannot trivially
                # predict region from coords. We'll still compute the
                # true geo embedding separately and attach it to outputs
                # for distance loss calculation.
                coords = batch['coords'].to(self.device)
                coords_zero = torch.zeros_like(coords)

                outputs = self.model(
                    input_values=batch['input_values'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    coordinates=coords_zero
                )

                # attach true geo embedding for use by the distance loss
                try:
                    outputs['true_geo_embedding'] = self.model.geo_embedding(coords)
                except Exception:
                    # if model doesn't expose geo_embedding for some reason,
                    # leave it as-is and let the loss function handle None
                    pass
                
                # criterion returns: (batch_total_loss, region_loss, gender_loss, distance_loss)
                batch_total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                    outputs,
                    batch['region_labels'].to(self.device),
                    batch['gender_labels'].to(self.device)
                )
                
                # quick checks for problematic values (NaN/inf/zero)
                if torch.isnan(batch_total_loss) or torch.isinf(batch_total_loss):
                    raise RuntimeError(f"Invalid loss (NaN/Inf) encountered: {batch_total_loss}")

                if batch_total_loss.item() == 0.0:
                    # print debug info to help trace why a loss would be exactly zero
                    warnings.warn(
                        "Batch loss is exactly 0.0 — dumping debug info (shapes, sample logits/labels)"
                    )
                    try:
                        rl = outputs['region_logits']
                        gl = outputs['gender_logits']
                        print('DEBUG region_logits.shape:', rl.shape)
                        print('DEBUG gender_logits.shape:', gl.shape)
                        print('DEBUG region_labels shape:', batch['region_labels'].shape)
                        print('DEBUG region_labels unique:', torch.unique(batch['region_labels']))
                        print('DEBUG region_logits sample (first row):', rl[0].detach().cpu().numpy())
                    except Exception:
                        print('DEBUG: failed to dump logits/labels')

                # scale for gradient accumulation
                loss = batch_total_loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # accumulate the real batch loss (not the scaled loss)
            epoch_loss += batch_total_loss.item()
            
            # Progress bar update
            # show running average loss
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'r_loss': f"{region_loss.item():.4f}",
                'g_loss': f"{gender_loss.item():.4f}",
                'd_loss': f"{distance_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 체크포인트 저장
            if self.global_step % self.save_steps == 0:
                metrics = {'train_loss': epoch_loss / (batch_idx + 1)}
                self.save_checkpoint(epoch, metrics, is_best=False)
        
        return {'train_loss': epoch_loss / len(self.train_loader)}
    
    def validate(self):
        """검증"""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                with autocast('cuda', dtype=self.amp_dtype):
                    coords = batch['coords'].to(self.device)
                    coords_zero = torch.zeros_like(coords)

                    outputs = self.model(
                        input_values=batch['input_values'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device),
                        coordinates=coords_zero
                    )

                    try:
                        outputs['true_geo_embedding'] = self.model.geo_embedding(coords)
                    except Exception:
                        pass
                    
                    # criterion returns: (batch_total_loss, region_loss, gender_loss, distance_loss)
                    batch_total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                        outputs,
                        batch['region_labels'].to(self.device),
                        batch['gender_labels'].to(self.device)
                    )
                epoch_loss += batch_total_loss.item()
                
                preds = outputs['region_logits'].argmax(dim=1)
                # compare
                lbls = batch['region_labels'].to(self.device)
                batch_correct = (preds == lbls).sum().item()
                correct += batch_correct
                # If a whole batch is correct (possible 1.0 accuracy), dump debug info
                if batch_correct == lbls.size(0):
                    warnings.warn('Validation batch fully correct — dumping debug info')
                    try:
                        print('DEBUG_VAL batch_size:', lbls.size(0))
                        print('DEBUG_VAL preds sample:', preds[:min(8, lbls.size(0))].cpu().numpy())
                        print('DEBUG_VAL labels sample:', lbls[:min(8, lbls.size(0))].cpu().numpy())
                        print('DEBUG_VAL logits sample row:', outputs['region_logits'][0].detach().cpu().numpy())
                    except Exception:
                        print('DEBUG_VAL: failed to dump batch info')
                total += lbls.size(0)  # ← 수정!
        
        return {
            'val_loss': epoch_loss / len(self.val_loader),
            'val_accuracy': correct / total
        }
    
    def train(self):
        """전체 학습 루프"""
        print("\n🚀 Starting training...")
        print(f"   Start epoch: {self.start_epoch}")
        print(f"   Total epochs: {self.num_epochs}")
        print(f"   Global step: {self.global_step}")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            
            is_best = val_metrics['val_accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['val_accuracy']
                print(f"  🎉 New best accuracy: {self.best_accuracy:.4f}")
            
            metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, metrics, is_best=is_best)
            
            if val_metrics['val_loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  ⚠️ No improvement for {self.patience_counter} epoch(s)")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
                break
        
        print("\n✅ Training completed!")
        print(f"   Best accuracy: {self.best_accuracy:.4f}")