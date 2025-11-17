"""
model_config.py
GeoAccentClassifier 모델 설정
"""

# ======================== 모델 아키텍처 ========================
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

# Task Configuration
NUM_REGIONS = 6
NUM_GENDERS = 2

# Architecture Dimensions
HIDDEN_DIM = 1024  # Wav2Vec2 XLSR-53 hidden size
GEO_EMBEDDING_DIM = 256
FUSION_DIM = 512

# Regularization
DROPOUT = 0.1

# Fine-tuning Strategy
FREEZE_LOWER_LAYERS = True
NUM_FROZEN_LAYERS = 16  # 하위 16개 레이어 freeze (총 24개 중)

# ======================== Loss Weights ========================
REGION_WEIGHT = 1.0
GENDER_WEIGHT = 0.3
DISTANCE_WEIGHT = 0.5

# ======================== Training Configuration ========================
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_WORKERS = 4

# Scheduler
SCHEDULER_TYPE = "cosine"  # 'cosine' or 'linear'

# ======================== Checkpoint & Logging ========================
CHECKPOINT_DIR = "./checkpoints_geo_accent"
LOG_DIR = "./logs_geo_accent"

# ======================== 사전 정의된 설정 (실험용) ========================

# Small Model (빠른 실험)
SMALL_CONFIG = {
    "model_name": "facebook/wav2vec2-base",
    "hidden_dim": 768,
    "geo_embedding_dim": 128,
    "fusion_dim": 256,
    "num_frozen_layers": 8
}

# Large Model (최고 성능)
LARGE_CONFIG = {
    "model_name": "facebook/wav2vec2-large-xlsr-53",
    "hidden_dim": 1024,
    "geo_embedding_dim": 512,
    "fusion_dim": 768,
    "num_frozen_layers": 20,
    "dropout": 0.2
}

# Full Fine-tune
FULL_FINETUNE_CONFIG = {
    "freeze_lower_layers": False,
    "num_frozen_layers": 0,
    "dropout": 0.15
}