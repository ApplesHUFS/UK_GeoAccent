"""
main.py
Main execution file for GeoAccent project
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
        'preprocess', help='Split dataset into train/validation/test'
    )
    preprocess_parser.add_argument('--dataset_name', type=str,
                                   default='ylacombe/english_dialects',
                                   help='HuggingFace dataset name')
    preprocess_parser.add_argument('--save_dir', type=str,
                                   default='./data/english_dialects',
                                   help='Directory to save split datasets')
    preprocess_parser.add_argument('--train_ratio', type=float,
                                   default=0.8, help='Training data ratio')
    preprocess_parser.add_argument('--val_ratio', type=float,
                                   default=0.1, help='Validation data ratio')
    preprocess_parser.add_argument('--test_ratio', type=float,
                                   default=0.1, help='Test data ratio')
    preprocess_parser.add_argument('--seed', type=int,
                                   default=42, help='Random seed')

    # 2. Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model_name', type=str,
                              default='facebook/wav2vec2-large-xlsr-53',
                              help='Pretrained Wav2Vec2 model name')
    train_parser.add_argument('--num_frozen_layers', type=int,
                              default=0, help='Number of lower Wav2Vec2 layers to freeze')
    train_parser.add_argument('--geo_embedding_dim', type=int,
                              default=256, help='Geographic embedding dimension')
    train_parser.add_argument('--fusion_dim', type=int,
                              default=512, help='Attention fusion dimension')
    train_parser.add_argument('--dropout', type=float,
                              default=0.1, help='Dropout rate')
    train_parser.add_argument('--use_fusion', action='store_true',
                              default=False, help='Use GeoEmbedding Fusion Module')

    train_parser.add_argument('--batch_size', type=int,
                              default=8, help='Batch size')
    train_parser.add_argument('--gradient_accumulation_steps', type=int,
                              default=2, help='Number of gradient accumulation steps')
    train_parser.add_argument('--learning_rate', type=float,
                              default=1e-5, help='Learning rate')
    train_parser.add_argument('--num_epochs', type=int,
                              default=40, help='Number of epochs')
    train_parser.add_argument('--num_workers', type=int,
                              default=4, help='Number of DataLoader workers')

    train_parser.add_argument('--region_weight', type=float,
                              default=1.0, help='Region classification loss weight')
    train_parser.add_argument('--gender_weight', type=float,
                              default=0.1, help='Gender classification loss weight')
    train_parser.add_argument('--distance_weight', type=float,
                              default=0.05, help='Distance regularization loss weight')

    train_parser.add_argument('--use_amp', action='store_true',
                              default=True, help='Use mixed precision')
    train_parser.add_argument('--max_grad_norm', type=float,
                              default=1.0, help='Gradient clipping norm')
    train_parser.add_argument('--warmup_steps', type=int,
                              default=500, help='Learning rate warmup steps')

    train_parser.add_argument('--data_dir', type=str,
                              default='./data/english_dialects',
                              help='Directory containing train/val/test datasets')
    train_parser.add_argument('--use_augment', action='store_true',
                              help='Use data augmentation')

    train_parser.add_argument('--checkpoint_dir', type=str,
                              default='./checkpoints', help='Directory to save checkpoints')
    train_parser.add_argument('--log_dir', type=str,
                              default='./logs', help='Directory to save logs')
    train_parser.add_argument('--save_steps', type=int,
                              default=500, help='Checkpoint saving frequency (steps)')
    train_parser.add_argument('--eval_steps', type=int,
                              default=500, help='Evaluation frequency (steps)')
    train_parser.add_argument('--resume', type=str,
                              default=None,
                              help='Path to checkpoint for resuming training')

    train_parser.add_argument('--early_stopping_patience', type=int,
                              default=8, help='Early stopping patience')
    train_parser.add_argument('--min_delta', type=float,
                              default=0.001, help='Minimum improvement for progress')

    train_parser.add_argument('--device', type=str,
                              default='cuda', help='Training device (cuda/cpu)')

    train_parser.add_argument('--config', type=str,
                              default=None, help='Optional JSON config file')

    # 3. Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str,
                             required=True, help='Checkpoint for evaluation')
    eval_parser.add_argument('--split', type=str,
                             default='test', choices=['validation', 'test'],
                             help='Dataset split to use')
    eval_parser.add_argument('--batch_size', type=int,
                             default=8, help='Batch size for evaluation')
    eval_parser.add_argument('--output_dir', type=str,
                             default='./results', help='Directory to save results')
    eval_parser.add_argument('--device', type=str,
                             default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Command execution
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
