import os
import random
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict


def split_dataset(dataset_name, configs, save_dir,
                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split dataset by speaker (no speaker overlap between splits)
    targeting ~8:1:1 sample ratio.
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Load datasets and add config_name column
    all_datasets = []
    for cfg in configs:
        print(f"Loading {cfg}...")
        # seed 설정
        ds = load_dataset(dataset_name, cfg, split="train", seed=random_seed) 
        ds = ds.add_column("config_name", [cfg] * len(ds)) 
        all_datasets.append(ds)
    
    full_dataset = concatenate_datasets(all_datasets)
    
    # 2. Determine speaker column
    speaker_col = "speaker_id" if "speaker_id" in full_dataset.column_names else "client_id"
    
    # 3. Group by config_name -> speaker -> indices
    config_to_speakers = defaultdict(lambda: defaultdict(list))
    for idx, sample in enumerate(full_dataset):
        config_name = sample["config_name"]
        speaker = sample[speaker_col]
        config_to_speakers[config_name][speaker].append(idx)
    
    # 4. Split by speakers (no overlap)
    train_idx, val_idx, test_idx = [], [], []
    config_speaker_stats = defaultdict(lambda: defaultdict(int))
    config_sample_stats = defaultdict(lambda: defaultdict(int))
    
    random.seed(random_seed) # Speaker shuffle을 위한 시드 설정
    
    for config_name, sp_dict in config_to_speakers.items():
        speakers = list(sp_dict.keys())
        random.shuffle(speakers)
        
        total_speakers = len(speakers)
        
        # Calculate speaker counts for 8:1:1 ratio
        n_train = max(1, int(total_speakers * train_ratio))
        n_val = max(1, int(total_speakers * val_ratio))
        n_test = max(1, total_speakers - n_train - n_val)
    
        # 합이 total_speakers를 초과하면 조정
        total_assigned = n_train + n_val + n_test
        if total_assigned > total_speakers:
            diff = total_assigned - total_speakers
            # test부터 차감
            if n_test > diff:
                n_test -= diff
            elif n_val > diff:
                n_val -= diff
            else: # n_train에서 차감
                n_train -= diff
        
        # Assign speakers to splits (NO OVERLAP)
        idx_ptr = 0
        
        # Train speakers
        for _ in range(n_train):
            if idx_ptr >= len(speakers): break
            sp = speakers[idx_ptr]
            train_idx.extend(sp_dict[sp])
            config_speaker_stats[config_name]['train'] += 1
            config_sample_stats[config_name]['train'] += len(sp_dict[sp])
            idx_ptr += 1
    
        # Val speakers
        for _ in range(n_val):
            if idx_ptr >= len(speakers): break
            sp = speakers[idx_ptr]
            val_idx.extend(sp_dict[sp])
            config_speaker_stats[config_name]['val'] += 1
            config_sample_stats[config_name]['val'] += len(sp_dict[sp])
            idx_ptr += 1
        
        # Test speakers
        for _ in range(n_test):
            if idx_ptr >= len(speakers): break
            sp = speakers[idx_ptr]
            test_idx.extend(sp_dict[sp])
            config_speaker_stats[config_name]['test'] += 1
            config_sample_stats[config_name]['test'] += len(sp_dict[sp])
            idx_ptr += 1
    
    # 5. Create final datasets (인덱스 리스트를 섞는 것은 데이터셋에 반영되지 않으므로 제거)
    train_ds = full_dataset.select(train_idx)
    val_ds = full_dataset.select(val_idx)
    test_ds = full_dataset.select(test_idx)

    
    # 6. Save to disk
    train_ds.save_to_disk(f"{save_dir}/train")
    val_ds.save_to_disk(f"{save_dir}/validation")
    test_ds.save_to_disk(f"{save_dir}/test")
    
    # 7. Print stats (출력 로직 생략 없이 유지)
    print("\n" + "="*70)
    print("Config-wise speaker distribution (NO OVERLAP):")
    print("="*70)
    for config_name in sorted(config_to_speakers.keys()):
        sp_stats = config_speaker_stats[config_name]
        sm_stats = config_sample_stats[config_name]
        total_sp = sum(sp_stats.values())
        total_sm = sum(sm_stats.values())
        print(f"\n[{config_name}]")
        print(f"  Speakers: train={sp_stats['train']}, val={sp_stats['val']}, "
              f"test={sp_stats['test']} (total={total_sp})")
        print(f"  Samples:  train={sm_stats['train']}, val={sm_stats['val']}, "
              f"test={sm_stats['test']} (total={total_sm})")
    
    print("\n" + "="*70)
    print("Global split summary:")
    print("="*70)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"Train: {len(train_ds):>6} ({len(train_ds)/total*100:5.1f}%)")
    print(f"Val:   {len(val_ds):>6} ({len(val_ds)/total*100:5.1f}%)")
    print(f"Test:  {len(test_ds):>6} ({len(test_ds)/total*100:5.1f}%)")
    print(f"Total: {total:>6}")
    print("="*70)
    
    return train_ds, val_ds, test_ds, config_speaker_stats, config_sample_stats


# Usage (예시 코드는 파일 실행 시점에 실행되도록 유지)
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
        random_seed=42,
    )