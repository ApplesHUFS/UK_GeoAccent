"""
preprocess.py
Split dataset into train/validation/test sets
"""

import shutil
import os
from datasets import load_dataset, concatenate_datasets

def split_dataset(
    dataset_name,
    configs,
    save_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "The sum of ratios must be 1.0"

    # Initialize save directory (optional: create if not exists)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # (1) Load datasets for each config, add label, and combine
    all_datasets = []
    for config in configs:
        try:
            print(f"[{config}] Loading dataset...")
            # Load dataset
            ds = load_dataset(dataset_name, config, split="train")
            
            # Add column with config name
            ds = ds.add_column("config_name", [config] * len(ds))
            
            # Optional: remove unnecessary columns to reduce size (keep only audio and label)
            # keep_cols = ['audio', 'config_name']
            # ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

            all_datasets.append(ds)
            print(f"   -> Loaded {len(ds)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load [{config}]: {e}")

    # Merge datasets
    full_dataset = concatenate_datasets(all_datasets)
    print(f"\nTotal number of samples: {len(full_dataset)}")

    # (2) Split Train vs Temp (Train: 80%, Temp: 20% for Val+Test)
    split_temp = full_dataset.train_test_split(
        test_size=1-train_ratio, seed=random_seed, shuffle=True
    )
    train_ds = split_temp['train']
    rest_ds = split_temp['test']
    
    print(f"Saving {len(train_ds)} training samples...")
    train_ds.save_to_disk(f"{save_dir}/train")

    # Clean up memory
    del train_ds
    import gc; gc.collect()

    # (3) Split Temp into Validation/Test according to ratio
    remaining_ratio = val_ratio + test_ratio
    test_size_real = test_ratio / remaining_ratio  # e.g., 0.1 / 0.2 = 0.5

    split_temp2 = rest_ds.train_test_split(
        test_size=test_size_real, seed=random_seed, shuffle=True
    )

    val_ds = split_temp2['train']
    test_ds = split_temp2['test']

    print(f"Saving {len(val_ds)} validation samples...")
    val_ds.save_to_disk(f"{save_dir}/validation")
    
    print(f"Saving {len(test_ds)} test samples...")
    test_ds.save_to_disk(f"{save_dir}/test")

    # Clear cache if needed
    # shutil.rmtree("/root/.cache/huggingface/datasets", ignore_errors=True)

    print("\n✅ Preprocessing completed: dataset with 'config_name' column has been saved.")


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
        train_ratio=0.8,  # Commonly used 8:1:1 split
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
