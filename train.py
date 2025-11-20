"""
train.py
모델 학습을 위한 메인 스크립트 - RTX 4090 최적화
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
import os

from models import GeoAccentClassifier, MultiTaskLossWithDistance
from trainer import GeoAccentTrainer
from data.dataset import EnglishDialectsDataset, collate_fn
from data.data_config import REGION_COORDS, REGION_LABELS


def parse_args():
    """Command line arguments 파싱"""
    parser = argparse.ArgumentParser(description='Train Geo-Accent Classifier')
    
    # Model
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--num_frozen_layers', type=int, default=16)
    parser.add_argument('--geo_embedding_dim', type=int, default=256)
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=500)
    
    # Early Stopping
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.001)
    
    # Checkpointing
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_geo_accent')
    parser.add_argument('--log_dir', type=str, default='./logs_geo_accent')
    
    # Loss weights
    parser.add_argument('--region_weight', type=float, default=1.0)
    parser.add_argument('--gender_weight', type=float, default=0.3)
    parser.add_argument('--distance_weight', type=float, default=0.5)
    
    # Data
    parser.add_argument('--use_augment', action='store_true', default=True)
    parser.add_argument('--max_audio_length', type=float, default=20.0)
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='geo-accent-classifier')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode (2 epochs, small data)')
    
    return parser.parse_args()


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_gpu_info():
    """GPU 정보 출력"""
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("GPU Information:")
        print("="*70)
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Available Memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
        print(f"  Reserved Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print("="*70 + "\n")
    else:
        print("\n⚠️  CUDA not available, using CPU\n")


def estimate_training_time(num_samples, batch_size, grad_accum_steps, num_epochs):
    """학습 시간 예측"""
    effective_batch = batch_size * grad_accum_steps
    steps_per_epoch = num_samples // effective_batch
    total_steps = steps_per_epoch * num_epochs
    
    # RTX 4090 기준 예상 (1.5초/스텝)
    estimated_seconds = total_steps * 1.5
    estimated_hours = estimated_seconds / 3600
    
    print("\n" + "="*70)
    print("Training Time Estimation (RTX 4090 기준):")
    print("="*70)
    print(f"  Samples: {num_samples}")
    print(f"  Effective Batch Size: {effective_batch}")
    print(f"  Steps per Epoch: {steps_per_epoch}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Estimated Time: {estimated_hours:.2f} hours")
    print(f"  Estimated Cost (@$0.59/hour): ${estimated_hours * 0.59:.2f}")
    print("="*70 + "\n")


def main():
    """메인 학습 함수"""
    args = parse_args()
    
    # Quick test mode
    if args.quick_test:
        print("🚀 Quick Test Mode Activated!")
        args.num_epochs = 2
        args.batch_size = 4
        args.gradient_accumulation_steps = 2
        args.save_steps = 100
        args.eval_steps = 100
        args.warmup_steps = 50
    
    # 시드 설정
    set_seed(args.seed)
    
    # GPU 정보 출력
    print_gpu_info()
    
    # Configuration 출력
    print("="*70)
    print("Training Configuration:")
    print("="*70)
    config_dict = vars(args)
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print("="*70 + "\n")
    
    # Config 저장
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ Config saved to {config_path}\n")
    
    # 데이터셋 및 데이터로더 생성
    print("📦 Loading datasets...")
    
    train_dataset = EnglishDialectsDataset(
        split='train',
        use_augment=args.use_augment,
        processor=None,
        max_audio_length=args.max_audio_length
    )
    
    val_dataset = EnglishDialectsDataset(
        split='validation',
        use_augment=False,
        processor=None,
        max_audio_length=args.max_audio_length
    )
    
    # Quick test mode: 데이터 축소
    if args.quick_test:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(100, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(50, len(val_dataset))))
        print("🔬 Using subset for quick testing")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Validation dataset: {len(val_dataset)} samples")
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}\n")
    
    # 학습 시간 예측
    estimate_training_time(
        num_samples=len(train_dataset),
        batch_size=args.batch_size,
        grad_accum_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs
    )
    
    # Model 인스턴스 생성
    print("🏗️  Building model...")
    model = GeoAccentClassifier(
        model_name=args.model_name,
        num_regions=len(REGION_LABELS),
        num_genders=2,
        geo_embedding_dim=args.geo_embedding_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        freeze_lower_layers=True,
        num_frozen_layers=args.num_frozen_layers
    )
    
    # 모델 파라미터 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"✅ Model created!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print()
    
    # Loss function
    criterion = MultiTaskLossWithDistance(
        region_weight=args.region_weight,
        gender_weight=args.gender_weight,
        distance_weight=args.distance_weight,
        distance_metric='cosine'
    )
    
    print("✅ Loss function initialized!\n")
    
    # Weights & Biases 초기화
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"xlsr53_{args.num_epochs}ep_bs{args.batch_size}x{args.gradient_accumulation_steps}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=config_dict
            )
            print("✅ Weights & Biases initialized!\n")
        except Exception as e:
            print(f"⚠️  Failed to initialize W&B: {e}")
            args.use_wandb = False
    
    # Trainer 인스턴스화
    print("🚀 Initializing trainer...")
    trainer = GeoAccentTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        region_coords=REGION_COORDS,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb
    )
    
    print("✅ Trainer initialized!\n")
    
    # 메모리 정보 출력 (학습 전)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"📊 Pre-training GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
    
    # 학습 시작
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.history['train_total_loss'].__len__(),
            val_acc=trainer.best_val_acc,
            is_best=False
        )
        print("✅ Checkpoint saved!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # 메모리 정보 출력 (학습 후)
    if torch.cuda.is_available():
        print(f"\n📊 Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"📊 Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
    
    # W&B 종료
    if args.use_wandb:
        import wandb
        wandb.finish()
    
    print("\n🎉 Training completed successfully!")
    print(f"💾 Checkpoints saved to: {args.checkpoint_dir}")
    print(f"📊 Logs saved to: {args.log_dir}\n")


if __name__ == "__main__":
    main()