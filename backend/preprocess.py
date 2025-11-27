import os
import random
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict

def split_dataset(dataset_name, configs, save_dir,
                                              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Load datasets and add region column
    all_datasets = []
    for cfg in configs:
        region = cfg.split("_")[0]
        ds = load_dataset(dataset_name, cfg, split="train")
        ds = ds.add_column("region", [region]*len(ds))
        all_datasets.append(ds)
    
    full_dataset = concatenate_datasets(all_datasets)
    
    # 2. Determine speaker column
    speaker_col = "speaker_id" if "speaker_id" in full_dataset.column_names else "client_id"
    
    # 3. Group by region -> speaker -> indices
    region_to_speakers = defaultdict(lambda: defaultdict(list))
    for idx, sample in enumerate(full_dataset):
        region = sample["region"]
        speaker = sample[speaker_col]
        region_to_speakers[region][speaker].append(idx)
    
    # 4. Prepare splits
    train_idx, val_idx, test_idx = [], [], []
    split_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    
    # Stats: speaker count & sample count per region per split
    region_speaker_stats = defaultdict(lambda: defaultdict(int))  # region -> split -> speaker count
    region_sample_stats  = defaultdict(lambda: defaultdict(int))  # region -> split -> sample count
    
    split_lists = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    # 4-1. Ensure each region has at least 1 speaker in each split
    for region, sp_dict in region_to_speakers.items():
        speakers = list(sp_dict.keys())
        random.shuffle(speakers)
        
        n_splits = len(split_lists)
        for i, split_name in enumerate(split_lists):
            if i < len(speakers):
                sp = speakers[i]
                split_lists[split_name].extend(sp_dict[sp])
                region_speaker_stats[region][split_name] += 1
                region_sample_stats[region][split_name] += len(sp_dict[sp])
        
        # 4-2. Remaining speakers assigned randomly according to split_ratios
        remaining_speakers = speakers[n_splits:]
        for sp in remaining_speakers:
            split_name = random.choices(list(split_ratios.keys()), weights=list(split_ratios.values()))[0]
            split_lists[split_name].extend(sp_dict[sp])
            region_speaker_stats[region][split_name] += 1
            region_sample_stats[region][split_name] += len(sp_dict[sp])
    
    # 5. Create final datasets
    train_ds = full_dataset.select(train_idx)
    val_ds   = full_dataset.select(val_idx)
    test_ds  = full_dataset.select(test_idx)
    
    # 6. Save to disk
    train_ds.save_to_disk(f"{save_dir}/train")
    val_ds.save_to_disk(f"{save_dir}/validation")
    test_ds.save_to_disk(f"{save_dir}/test")
    
    # 7. Print stats
    print("Region-wise speaker distribution:")
    for region in region_to_speakers:
        sp_stats = region_speaker_stats[region]
        sm_stats = region_sample_stats[region]
        total_sp = sum(sp_stats.values())
        total_sm = sum(sm_stats.values())
        print(f"[{region}] speakers: train={sp_stats['train']}, val={sp_stats['val']}, test={sp_stats['test']} (total={total_sp}), "
              f"samples: train={sm_stats['train']}, val={sm_stats['val']}, test={sm_stats['test']} (total={total_sm})")
    
    print("\nGlobal split counts (samples):")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds, region_speaker_stats, region_sample_stats


# Usage example
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
        test_ratio=0.1
    )
