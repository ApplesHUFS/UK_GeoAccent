"""
model_config.py
GeoAccentClassifier 모델 설정 - RTX 4090 24GB 최적화
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
# RTX 4090 24GB 최적화 설정
LEARNING_RATE = 1e-5
BATCH_SIZE = 4  # 4로 시작 (안정성 우선)
GRADIENT_ACCUMULATION_STEPS = 4  # 실질적 배치 = 16
NUM_EPOCHS = 25  # 31시간 데이터에 적합
NUM_WORKERS = 4  # 6 vCPU 중 4개 활용

# Mixed Precision Training (속도 30% 향상)
USE_AMP = True  # Automatic Mixed Precision
AMP_DTYPE = "bfloat16"  # RTX 4090은 bf16 지원 (fp16보다 안정적)

# Gradient Clipping
MAX_GRAD_NORM = 1.0

# Scheduler
SCHEDULER_TYPE = "cosine"  # 'cosine' or 'linear'
WARMUP_STEPS = 500  # 초기 학습 안정화

# Early Stopping (조기 종료로 시간 절약)
EARLY_STOPPING_PATIENCE = 5  # 5 epoch 동안 개선 없으면 종료
MIN_DELTA = 0.001  # 최소 개선 폭

# ======================== Checkpoint & Logging ========================
CHECKPOINT_DIR = "./checkpoints_geo_accent"
LOG_DIR = "./logs_geo_accent"

# Checkpoint 저장 전략
SAVE_STRATEGY = "steps"  # 'epoch' or 'steps'
SAVE_STEPS = 500  # 500 스텝마다 저장 (Spot Instance 대비)
SAVE_TOTAL_LIMIT = 3  # 최근 3개 체크포인트만 유지 (디스크 절약)

# Logging
LOGGING_STEPS = 50  # 50 스텝마다 로그
EVAL_STEPS = 500  # 500 스텝마다 검증

# ======================== 메모리 최적화 ========================

# DataLoader 최적화
PIN_MEMORY = True  # GPU 전송 속도 향상
PERSISTENT_WORKERS = True  # Worker 재사용으로 오버헤드 감소

# ======================== 실험 추적 ========================
# Weights & Biases 통합 (선택사항)
USE_WANDB = False  # True로 설정하면 W&B 로깅
WANDB_PROJECT = "geo-accent-classifier"
WANDB_RUN_NAME = "xlsr53-31h-rtx4090"
