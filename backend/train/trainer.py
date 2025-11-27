"""
train/trainer.py
Trainer for GeoAccent
"""
import os
import glob
import logging
import torch
import contextlib
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def setup_file_logging(log_dir):
    """Set up logging to file ONLY, using a specific logger name."""
    logger = logging.getLogger('FileLogger') # 전용 로거 이름 사용
    logger.setLevel(logging.INFO)
    
    # 중복 핸들러 방지
    if not logger.handlers:
        log_file = os.path.join(log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        # 시간, 메시지 포맷 유지
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

class AccentTrainer:
    """
    Trainer for GeoAccent Classifier
    """

    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        region_coords,
        device='cuda',
        learning_rate=5e-6,
        num_epochs=40,
        gradient_accumulation_steps=2,
        use_amp=True,
        max_grad_norm=1.0,
        warmup_steps=500,
        early_stopping_patience=8,
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

        # Training parameters
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

        # Logging & checkpointing
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.file_logger = setup_file_logging(log_dir)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # AMP
        self.scaler = GradScaler() if use_amp else None
        self.amp_dtype = torch.bfloat16 if use_amp else torch.float32 

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save a checkpoint"""
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
                print(f'Saved best checkpoint (epoch {epoch}, val_acc={metrics.get("val_accuracy", 0):.4f})')
            else:
                last_path = os.path.join(self.checkpoint_dir, 'last.pt')
                if os.path.exists(last_path):
                    os.remove(last_path)
                os.rename(temp_path, last_path)
                print(f'Saved last checkpoint (epoch {epoch})')

            self._cleanup_old_checkpoints(keep_last_n=1)
        except RuntimeError as e:
            print(f"Checkpoint save failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self._cleanup_old_checkpoints(keep_last_n=0)
            try:
                torch.save(checkpoint, temp_path)
                last_path = os.path.join(self.checkpoint_dir, 'last.pt')
                os.rename(temp_path, last_path)
                print("Retry successful")
            except Exception as e2:
                print(f"Retry failed: {e2}, continuing without saving checkpoint.")

    def _cleanup_old_checkpoints(self, keep_last_n=1):
        """Remove old checkpoints"""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')
        checkpoints = sorted(glob.glob(pattern))
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints:
                try:
                    os.remove(old_ckpt)
                except Exception as e:
                    print(f'Failed to remove {old_ckpt}: {e}')

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}, best_accuracy={self.best_accuracy:.4f}")
        return checkpoint

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            if self.use_amp:
                amp_context = autocast(dtype=self.amp_dtype)
            else:
                amp_context = contextlib.nullcontext()

            with amp_context:
                # use_fusion 여부에 따라 coordinates 전달 결정
                coords = batch['coords'].to(self.device)

                # use_fusion=True면 실제 좌표, False면 None
                input_coords = coords if self.model.use_fusion else None

                outputs = self.model(
                    input_values=batch['input_values'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    coordinates=input_coords,
                )

                # distance loss를 위한 true geo embedding (use_fusion=True일 때만)
                if self.model.use_fusion and self.model.geo_embedding is not None:
                    try:
                        outputs['true_geo_embedding'] = self.model.geo_embedding(coords)
                    except Exception:
                        pass

                batch_total_loss, region_loss, gender_loss, distance_loss = self.criterion(
                    outputs,
                    batch['region_labels'].to(self.device),
                    batch['gender_labels'].to(self.device),
                )

                if torch.isnan(batch_total_loss) or torch.isinf(batch_total_loss):
                    raise RuntimeError(f"Invalid loss encountered: {batch_total_loss}")

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

            epoch_loss += batch_total_loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'r_loss': f"{region_loss.item():.4f}",
                'g_loss': f"{gender_loss.item():.4f}",
                'd_loss': f"{distance_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            if self.global_step % self.save_steps == 0:
                metrics = {'train_loss': avg_loss}
                self.save_checkpoint(epoch, metrics, is_best=False)
        

        return {'train_loss': epoch_loss / len(self.train_loader)}

    def validate(self):
        """Validation"""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                if self.use_amp:
                    amp_context = autocast(dtype=self.amp_dtype)
                else:
                    amp_context = contextlib.nullcontext()

                with amp_context:
                    coords = batch['coords'].to(self.device)
                    input_coords = coords if self.model.use_fusion else None

                    outputs = self.model(
                        input_values=batch['input_values'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device),
                        coordinates=input_coords,
                    )

                    if self.model.use_fusion and self.model.geo_embedding is not None:
                        try:
                            outputs['true_geo_embedding'] = self.model.geo_embedding(coords)
                        except Exception:
                            pass

                    batch_total_loss, _, _, _ = self.criterion(
                        outputs,
                        batch['region_labels'].to(self.device),
                        batch['gender_labels'].to(self.device),
                    )

                epoch_loss += batch_total_loss.item()
                preds = outputs['region_logits'].argmax(dim=1)
                lbls = batch['region_labels'].to(self.device)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        return {
            'val_loss': epoch_loss / len(self.val_loader),
            'val_accuracy': correct / total,
        }


    def train(self):
        """Full training loop"""
        start_msg = f"\nStarting training from epoch {self.start_epoch} for {self.num_epochs} epochs"
        print(start_msg)
        self.file_logger.info(start_msg)
        print("==================================================")
        self.file_logger.info("==================================================")

        for epoch in range(self.start_epoch, self.num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")

            self.file_logger.info(f"\nEpoch {epoch} Results:")
            self.file_logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            self.file_logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            self.file_logger.info(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")

            is_best = val_metrics['val_accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['val_accuracy']
                print(f"  🎉 New best accuracy: {self.best_accuracy:.4f}") 
                self.file_logger.info(f"  🎉 New best accuracy: {self.best_accuracy:.4f}") 

            metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, metrics, is_best=is_best)

            if val_metrics['val_loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                self.file_logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        final_msg = f"Training completed! Best accuracy: {self.best_accuracy:.4f}"
        print("\n==================================================")
        print(final_msg)
        print("==================================================")
        self.file_logger.info("\n==================================================")
        self.file_logger.info(final_msg)
        self.file_logger.info("==================================================")