"""
train.py
모델 학습을 위한 메인 스크립트
"""

import torch
from torch.utils.data import DataLoader

from models import GeoAccentClassifier, MultiTaskLossWithDistance
from trainer import GeoAccentTrainer
from data.dataset import EnglishDialectsDataset, collate_fn
from data.data_config import REGION_COORDS, REGION_LABELS


def main():
    """메인 학습 함수"""
    
    # Configuration
    config = {
        'model_name': 'facebook/wav2vec2-large-xlsr-53',
        'batch_size': 8,
        'learning_rate': 1e-5,
        'num_epochs': 30,
        'num_frozen_layers': 16,
        'geo_embedding_dim': 256,
        'fusion_dim': 512,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4
    }
    
    print("="*70)
    print("Configuration:")
    print("="*70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*70 + "\n")
    
    # 데이터셋 및 데이터로더 생성
    print("📦 Loading datasets...")
    train_dataset = EnglishDialectsDataset(
        split='train',
        use_augment=True,
        processor=None
    )
    
    val_dataset = EnglishDialectsDataset(
        split='validation',
        use_augment=False,
        processor=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Validation dataset: {len(val_dataset)} samples")
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}\n")
    
    # Model 인스턴스 생성
    print("🏗️  Building model...")
    model = GeoAccentClassifier(
        model_name=config['model_name'],
        num_regions=len(REGION_LABELS),
        num_genders=2,
        geo_embedding_dim=config['geo_embedding_dim'],
        fusion_dim=config['fusion_dim'],
        dropout=config['dropout'],
        freeze_lower_layers=True,
        num_frozen_layers=config['num_frozen_layers']
    )
    
    # Loss function
    criterion = MultiTaskLossWithDistance(
        region_weight=1.0,
        gender_weight=0.3,
        distance_weight=0.5,
        distance_metric='cosine'
    )
    
    print("✅ Model and loss function initialized!\n")
    
    # Trainer 인스턴스화
    print("🚀 Initializing trainer...")
    trainer = GeoAccentTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        region_coords=REGION_COORDS,
        device=config['device'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        checkpoint_dir='./checkpoints_geo_accent',
        log_dir='./logs_geo_accent'
    )
    
    print("✅ Trainer initialized!\n")
    
    # 학습 시작
    trainer.train()
    
    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main()