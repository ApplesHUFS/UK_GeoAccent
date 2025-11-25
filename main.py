"""
main.py
GeoAccent 프로젝트 메인 실행 파일
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="GeoAccent: Geographic-Aware British English Accent Classifier"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 1. Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='데이터셋을 train/validation/test로 분할'
    )
    
    preprocess_parser.add_argument(
        '--dataset_name',
        type=str,
        default='ylacombe/english_dialects',
        help='HuggingFace 데이터셋 이름'
    )
    
    preprocess_parser.add_argument(
        '--save_dir',
        type=str,
        default='./data/english_dialects',
        help='저장할 디렉토리'
    )
    
    preprocess_parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='훈련 데이터 비율'
    )
    
    preprocess_parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='검증 데이터 비율'
    )
    
    preprocess_parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='테스트 데이터 비율'
    )
    
    preprocess_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드'
    )
    
    # 2. Train command
    train_parser = subparsers.add_parser(
        'train',
        help='모델 훈련'
    )

    # Model args
    train_parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/wav2vec2-large-xlsr-53',
        help='Pretrained Wav2Vec2 모델 이름'
    )

    train_parser.add_argument(
        '--num_frozen_layers',
        type=int,
        default=16,
        help='Freeze할 Wav2Vec2 하위 레이어 수'
    )

    train_parser.add_argument(
        '--geo_embedding_dim',
        type=int,
        default=256,
        help='Geographic embedding 차원'
    )

    train_parser.add_argument(
        '--fusion_dim',
        type=int,
        default=512,
        help='Attention fusion 차원'
    )

    train_parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout 비율'
    )

    # Training args
    train_parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='배치 크기'
    )

    train_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help='Gradient accumulation 스텝'
    )

    train_parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='학습률'
    )

    train_parser.add_argument(
        '--num_epochs',
        type=int,
        default=25,
        help='에포크 수'
    )

    train_parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader worker 수'
    )

    # Loss weights
    train_parser.add_argument(
        '--region_weight',
        type=float,
        default=1.0,
        help='Region loss 가중치'
    )

    train_parser.add_argument(
        '--gender_weight',
        type=float,
        default=0.3,
        help='Gender loss 가중치'
    )

    train_parser.add_argument(
        '--distance_weight',
        type=float,
        default=0.5,
        help='Distance loss 가중치'
    )

    # Optimization
    train_parser.add_argument(
        '--use_amp',
        action='store_true',
        default=True,
        help='Mixed Precision 사용 여부'
    )

    train_parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Gradient clipping norm'
    )

    train_parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Learning rate warmup 스텝'
    )

    # Data
    train_parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/english_dialects',
        help='train/val/test 데이터셋 디렉토리'
    )

    train_parser.add_argument(
        '--use_augment',
        action='store_true',
        help='Augmentation 사용 여부'
    )

    # Checkpointing
    train_parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='체크포인트 저장 디렉토리'
    )

    train_parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='로그 저장 디렉토리'
    )

    train_parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='체크포인트 저장 주기 (steps)'
    )

    train_parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='검증 주기 (steps)'
    )

    # Early stopping
    train_parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )

    train_parser.add_argument(
        '--min_delta',
        type=float,
        default=0.001,
        help='Early stopping 최소 개선 폭'
    )

    # Device
    train_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='학습 디바이스 (cuda/cpu)'
    )

    # Wandb
    train_parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Weights & Biases 사용 여부'
    )

    train_parser.add_argument(
        '--wandb_project',
        type=str,
        default='geo-accent-classifier',
        help='W&B 프로젝트 이름'
    )

    train_parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='W&B run 이름'
    )

    # Config file (optional)
    train_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='설정 파일 경로 (JSON)'
    )

    
    # 3. Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='모델 평가'
    )
    
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='평가할 체크포인트 경로'
    )
    
    eval_parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['validation', 'test'],
        help='평가할 데이터 split'
    )
    
    eval_parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='배치 크기'
    )
    
    eval_parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='결과 저장 디렉토리'
    )
    
    args = parser.parse_args()
    
    # Command 실행
    if args.command == 'preprocess':
        from preprocess import split_dataset

        configs = [
        'irish_male', 'midlands_female', 'midlands_male', 'northern_female',
        'northern_male', 'scottish_female', 'scottish_male',
        'southern_female', 'southern_male', 'welsh_female', 'welsh_male'
    ]
        split_dataset(
            dataset_name=args.dataset_name,
            configs=configs, 
            save_dir=args.save_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
        
    elif args.command == 'train':
        from train.train import train_model
        train_model(args)
        
    elif args.command == 'evaluate':
        from evaluation.evaluate import evaluate_model
        evaluate_model(args)
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
