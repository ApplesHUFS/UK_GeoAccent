"""
preprocess.py
데이터셋을 train/validation/test로 분할
"""

import os
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split


def split_dataset(
    dataset_name="ylacombe/english_dialects",
    save_dir="./data/english_dialects",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    HuggingFace 데이터셋을 train/validation/test로 분할
    
    Args:
        dataset_name: HuggingFace 데이터셋 이름
        save_dir: 저장할 디렉토리
        train_ratio: 훈련 데이터 비율 (0.7 = 70%)
        val_ratio: 검증 데이터 비율 (0.15 = 15%)
        test_ratio: 테스트 데이터 비율 (0.15 = 15%)
        random_seed: 재현성을 위한 시드
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "비율의 합은 1.0이어야 합니다"
    
    print("=" * 50)
    print("데이터셋 분할 시작")
    print("=" * 50)
    
    # 1. 데이터셋 로드
    print(f"\n1. Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # 모든 split을 하나로 합치기
    if isinstance(dataset, DatasetDict):
        # 여러 split이 있으면 모두 합침
        all_data = []
        for split_name in dataset.keys():
            all_data.append(dataset[split_name])
        
        # concatenate
        from datasets import concatenate_datasets
        full_dataset = concatenate_datasets(all_data)
    else:
        full_dataset = dataset
    
    print(f"   전체 샘플 수: {len(full_dataset)}")
    
    # 2. Train / Temp 분할 (temp = val + test)
    print(f"\n2. Splitting dataset...")
    print(f"   Train: {train_ratio*100:.1f}%")
    print(f"   Validation: {val_ratio*100:.1f}%")
    print(f"   Test: {test_ratio*100:.1f}%")
    
    # 인덱스 생성
    indices = list(range(len(full_dataset)))
    
    # Train / Temp 분할
    temp_ratio = val_ratio + test_ratio
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    # Temp를 Validation / Test로 분할
    val_size_in_temp = val_ratio / temp_ratio
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size_in_temp),
        random_state=random_seed,
        shuffle=True
    )
    
    # 3. 데이터셋 분할 적용
    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)
    test_dataset = full_dataset.select(test_indices)
    
    print(f"\n   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # 4. 저장
    print(f"\n3. Saving to disk: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset.save_to_disk(f"{save_dir}/train")
    val_dataset.save_to_disk(f"{save_dir}/validation")
    test_dataset.save_to_disk(f"{save_dir}/test")
    
    print(f"   ✅ Train saved to: {save_dir}/train")
    print(f"   ✅ Validation saved to: {save_dir}/validation")
    print(f"   ✅ Test saved to: {save_dir}/test")
    
    print("\n" + "=" * 50)
    print("데이터셋 분할 완료!")
    print("=" * 50)
    
    return {
        'train': len(train_dataset),
        'validation': len(val_dataset),
        'test': len(test_dataset)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터셋을 train/val/test로 분할")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ylacombe/english_dialects",
        help="HuggingFace 데이터셋 이름"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/english_dialects",
        help="저장할 디렉토리"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="훈련 데이터 비율 (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="검증 데이터 비율 (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="테스트 데이터 비율 (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (default: 42)"
    )
    
    args = parser.parse_args()
    
    split_dataset(
        dataset_name=args.dataset_name,
        save_dir=args.save_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
