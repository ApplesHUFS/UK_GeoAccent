# dev_check.py

from data import EnglishDialectsDataset, collate_fn

# 데이터셋 초기화
ds = EnglishDialectsDataset(split='train', use_augment=False, processor=None)
print("✅ Dataset loaded successfully!")
print("Sample count:", len(ds))

# 첫 번째 샘플 출력 테스트
sample = ds[0]
print("Sample keys:", sample.keys())
