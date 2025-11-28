"""
preprocess.py
Split dataset into train/validation/test sets based on SPEAKER ID
"""

import os
import random
import shutil
from collections import defaultdict
from datasets import load_dataset, Dataset, concatenate_datasets

def split_dataset(
    dataset_name,
    configs,
    save_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 최종적으로 합칠 리스트
    final_train_list = []
    final_val_list = []
    final_test_list = []

    print(f"Processing stratified speaker split...")

    for config in configs:
        try:
            print(f"[{config}] Loading and splitting by speaker...")
            # 데이터셋 로드
            ds = load_dataset(dataset_name, config, split="train")
            
            # config_name 추가
            ds = ds.add_column("config_name", [config] * len(ds))

            # 1. 화자별로 데이터 그룹화
            speaker_dict = defaultdict(list)
            for item in ds:
                spk_id = item['speaker_id']
                speaker_dict[spk_id].append(item)
            
            speakers = list(speaker_dict.keys())
            
            # 2. 화자 섞기
            random.seed(random_seed)
            random.shuffle(speakers)
            
            # 3. 화자 단위 분할 지점 계산
            n_total_spk = len(speakers)
            n_train = int(n_total_spk * train_ratio)
            n_val = int(n_total_spk * val_ratio)
            
            # 최소 1명은 보장 (데이터가 너무 적은 경우를 대비)
            if n_train == 0 and n_total_spk > 0: n_train = 1
            
            train_spks = speakers[:n_train]
            val_spks = speakers[n_train:n_train+n_val]
            test_spks = speakers[n_train+n_val:]
            
            # 만약 val/test 화자가 0명이면 train에서 떼오거나 경고 (여기서는 단순하게 처리)
            if not val_spks and len(train_spks) > 1: 
                val_spks = [train_spks.pop()]
            if not test_spks and len(train_spks) > 1: 
                test_spks = [train_spks.pop()]

            # 4. 데이터 리스트에 추가
            for spk in train_spks:
                final_train_list.extend(speaker_dict[spk])
            for spk in val_spks:
                final_val_list.extend(speaker_dict[spk])
            for spk in test_spks:
                final_test_list.extend(speaker_dict[spk])
                
            print(f"   -> Speakers: Train={len(train_spks)}, Val={len(val_spks)}, Test={len(test_spks)}")

        except Exception as e:
            print(f"❌ Failed to process [{config}]: {e}")

    # 5. 리스트를 다시 HuggingFace Dataset으로 변환 및 저장
    print("\nConverting to Datasets and saving...")
    
    train_ds = Dataset.from_list(final_train_list)
    val_ds = Dataset.from_list(final_val_list)
    test_ds = Dataset.from_list(final_test_list)

    print(f"Saving Train: {len(train_ds)} samples")
    train_ds.save_to_disk(f"{save_dir}/train")
    
    print(f"Saving Validation: {len(val_ds)} samples")
    val_ds.save_to_disk(f"{save_dir}/validation")
    
    print(f"Saving Test: {len(test_ds)} samples")
    test_ds.save_to_disk(f"{save_dir}/test")

    print("\n✅ Speaker-Independent Preprocessing Completed!")

if __name__ == "__main__":
    configs = [
        'irish_male',
        'midlands_male', 'midlands_female', 
        'northern_male', 'northern_female', 
        'scottish_male', 'scottish_female', 
        'southern_male', 'southern_female', 
        'welsh_male', 'welsh_female'
    ]
    
    split_dataset(
        dataset_name="ylacombe/english_dialects",
        configs=configs,
        save_dir="./data/english_dialects",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )