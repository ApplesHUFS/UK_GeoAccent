import shutil
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict


def split_dataset(
    dataset_name,
    configs,
    save_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "비율 합이 1.0이어야 함"

    # (1) config별로 조금씩 불러와 통합
    all_datasets = []
    for config in configs:
        print(f"[{config}] 데이터셋 로딩")
        ds = load_dataset(dataset_name, config, split="train")
        all_datasets.append(ds)

    full_dataset = concatenate_datasets(all_datasets)

    # (2) Train split 만들고 저장
    split_temp = full_dataset.train_test_split(
        test_size=1-train_ratio, seed=random_seed, shuffle=True
    )
    train_ds, rest_ds = split_temp['train'], split_temp['test']
    print(f"Train: {len(train_ds)}개 저장")
    train_ds.save_to_disk(f"{save_dir}/train")

    del all_datasets, train_ds
    import gc; gc.collect()
    shutil.rmtree("/root/.cache/huggingface/datasets", ignore_errors=True)

    # (3) Validation split 만들고 저장
    split_temp2 = rest_ds.train_test_split(
        test_size=test_ratio/(val_ratio+test_ratio), seed=random_seed, shuffle=True
    )

    val_ds, test_ds = split_temp2['train'], split_temp2['test']
    print(f"Validation: {len(val_ds)}개 저장")
    val_ds.save_to_disk(f"{save_dir}/validation")
    del val_ds
    gc.collect()
    shutil.rmtree("/root/.cache/huggingface/datasets", ignore_errors=True)

    # (4) Test split 저장
    print(f"Test: {len(test_ds)}개 저장")
    test_ds.save_to_disk(f"{save_dir}/test")
    del test_ds
    gc.collect()
    shutil.rmtree("/root/.cache/huggingface/datasets", ignore_errors=True)

    print("\n✅ train/val/test 분할 및 저장, 단계별 캐시 삭제 완료")


if __name__ == "__main__":
    configs = [
        'irish_male', 'midlands_female', 'midlands_male', 'northern_female',
        'northern_male', 'scottish_female', 'scottish_male',
        'southern_female', 'southern_male', 'welsh_female', 'welsh_male'
    ]
    split_dataset(
        dataset_name="ylacombe/english_dialects",
        configs=configs,
        save_dir="./data/english_dialects",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
