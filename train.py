"""
train.py
모델 학습을 위한 메인 스크립트
"""

import torch
from torch.utils.data import DataLoader

from models import GeoAccentClassifier, MultiTaskLossWithDistance
from trainer import GeoAccentTrainer
# from data.dataset import EnglishDialectsDataset  # TODO: 실제 구현 후 언코멘트


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
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*70)
    print("Configuration:")
    print("="*70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*70 + "\n")
    
    # TODO: 실제 데이터로더 구현 후 교체
    # train_dataset = EnglishDialectsDataset(split='train', augment=True)
    # val_dataset = EnglishDialectsDataset(split='val', augment=False)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=4
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=4
    # )
    
    print("⚠️  Using dummy data loaders for testing")
    print("TODO: Replace with actual EnglishDialectsDataset from data/dataset.py\n")
    
    # Model 인스턴스 생성
    model = GeoAccentClassifier(
        model_name=config['model_name'],
        num_regions=6,
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
    
    # Trainer 인스턴스화 (TODO: 실제 loader로 교체)
    # trainer = GeoAccentTrainer(
    #     model=model,
    #     criterion=criterion,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     region_coords=REGION_COORDS,
    #     device=config['device'],
    #     learning_rate=config['learning_rate'],
    #     num_epochs=config['num_epochs'],
    #     checkpoint_dir='./checkpoints_geo_accent',
    #     log_dir='./logs_geo_accent'
    # )
    
    # trainer.train()
    
    print("✅ Model, Loss, and Trainer initialized successfully!")
    print("   Ready to train once data loaders are implemented.\n")


if __name__ == "__main__":
    main()
