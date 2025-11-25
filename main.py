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
    
    train_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='설정 파일 경로 (JSON)'
    )
    
    train_parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='배치 크기'
    )
    
    train_parser.add_argument(
        '--num_epochs',
        type=int,
        default=25,
        help='에포크 수'
    )
    
    train_parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='학습률'
    )
    
    train_parser.add_argument(
        '--use_augment',
        action='store_true',
        help='Augmentation 사용 여부'
    )
    
    train_parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='체크포인트 저장 디렉토리'
    )

    train_parser.add_argument(
    '--data_dir',
    type=str,
    default='./data/english_dialects',
    help='train/val/test 데이터셋이 들어있는 디렉토리'
)
    train_parser.add_argument(
    '--num_workers',
    type=int,
    default=4,
    help='DataLoader에서 사용할 worker(서브프로세스) 수'
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
