"""
train/train.py
Training script for GeoAccentClassifier
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os

from models import GeoAccentClassifier, MultiTaskLossWithDistance
from train.trainer import AccentTrainer
from data import EnglishDialectsDataset, collate_fn
from utils.config import REGION_COORDS, REGION_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description='Train Geo-Accent Classifier')

    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--num_frozen_layers', type=int, default=16)
    parser.add_argument('--geo_embedding_dim', type=int, default=256)
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_fusion', action='store_true', default=True)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--region_weight', type=float, default=1.0)
    parser.add_argument('--gender_weight', type=float, default=0.3)
    parser.add_argument('--distance_weight', type=float, default=0.5)

    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=500)

    parser.add_argument('--data_dir', type=str, default='./data/english_dialects')
    parser.add_argument('--use_augment', action='store_true')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.001)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='geo-accent-classifier')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    return parser.parse_args()


def train_model(args):
    print("=" * 50)
    print("GeoAccent Classifier Training")
    print("=" * 50)

    print("\n1. Loading datasets...")
    train_dataset = EnglishDialectsDataset(split='train', use_augment=args.use_augment, data_dir=args.data_dir)
    val_dataset = EnglishDialectsDataset(split='validation', use_augment=False, data_dir=args.data_dir)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    print("\n2. Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    print("\n3. Creating model...")
    model = GeoAccentClassifier(
        model_name=args.model_name,
        num_regions=len(REGION_LABELS),
        num_genders=2,
        hidden_dim=1024,
        geo_embedding_dim=args.geo_embedding_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        freeze_lower_layers=True,
        num_frozen_layers=args.num_frozen_layers
        use_fusion=args.use_fusion
    )
    model = model.to(args.device)
    print(f"   Model loaded on {args.device}")

    print("\n4. Creating loss function...")
    criterion = MultiTaskLossWithDistance(
        region_weight=args.region_weight,
        gender_weight=args.gender_weight,
        distance_weight=args.distance_weight
    )

    print("\n5. Creating trainer...")
    trainer = AccentTrainer(
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

    if args.resume:
        print(f"\n6. Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("\n7. Starting training...")
    print("=" * 50)
    trainer.train()

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
