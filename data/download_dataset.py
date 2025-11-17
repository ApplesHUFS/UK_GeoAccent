from datasets import load_dataset, concatenate_datasets
import os

def download_english_dialects():
    
    dataset_name = "ylacombe/english_dialects"
    
    configs = [
        "irish_male", "midlands_female", "midlands_male",
        "northern_female", "northern_male",
        "scottish_female", "scottish_male",
        "southern_female", "southern_male",
        "welsh_female", "welsh_male"
    ]
    
    save_dir = "./data/english_dialects"
    os.makedirs(save_dir, exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        print(f"\n📥 Downloading {split} split...")
        
        datasets_list = []
        for cfg in configs:
            print(f"  - Loading {cfg}...")
            ds_cfg = load_dataset(dataset_name, cfg, split=split)
            ds_cfg = ds_cfg.add_column("config_name", [cfg] * len(ds_cfg))
            datasets_list.append(ds_cfg)
        
        combined = concatenate_datasets(datasets_list)
        
        save_path = f"{save_dir}/{split}"
        combined.save_to_disk(save_path)
        
        print(f"✅ Saved {split}: {len(combined)} samples → {save_path}")
    
    print("\n🎉 All downloads complete!")
    print(f"📁 Dataset location: {save_dir}")


if __name__ == "__main__":
    download_english_dialects()
