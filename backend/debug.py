from datasets import load_from_disk
from collections import Counter

val_raw = load_from_disk('./data/english_dialects/validation')
config_counts = Counter([s['config_name'] for s in val_raw])
print("Val set config distribution:")
for cfg, count in sorted(config_counts.items()):
    print(f"  {cfg}: {count} samples")