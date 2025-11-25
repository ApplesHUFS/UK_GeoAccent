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
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "비율 합이 1.0이어야 함"

    # 저장 경로 초기화 (선택사항: 기존 폴더 있으면 삭제 후 재생성 방지하거나 경고)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # (1) config별로 불러와서 라벨 달고 통합
    all_datasets = []
    for config in configs:
        try:
            print(f"[{config}] 데이터셋 로딩 중...")
            # 데이터 로드
            ds = load_dataset(dataset_name, config, split="train")
            
            ds = ds.add_column("config_name", [config] * len(ds))
            
            # (선택) 불필요한 컬럼 제거하여 용량 줄이기 (오디오와 라벨만 남김)
            # 텍스트가 필요 없다면 주석 해제하여 사용하세요.
            # keep_cols = ['audio', 'config_name']
            # ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

            all_datasets.append(ds)
            print(f"   -> {len(ds)}개 로드 완료")
            
        except Exception as e:
            print(f"❌ [{config}] 로드 실패: {e}")

    # 데이터 병합
    full_dataset = concatenate_datasets(all_datasets)
    print(f"\n총 데이터 개수: {len(full_dataset)}")

    # (2) Train split (Train vs Temp)
    # 80% Train, 20% Temp (Val+Test)
    split_temp = full_dataset.train_test_split(
        test_size=1-train_ratio, seed=random_seed, shuffle=True
    )
    train_ds = split_temp['train']
    rest_ds = split_temp['test']
    
    print(f"Train: {len(train_ds)}개 저장 중...")
    train_ds.save_to_disk(f"{save_dir}/train")

    # 메모리 정리
    del train_ds
    import gc; gc.collect()

    # (3) Validation/Test split (Temp를 반으로 나눔)
    # 남은 20% 중에서 Val:Test 비율 계산
    remaining_ratio = val_ratio + test_ratio
    test_size_real = test_ratio / remaining_ratio  # 0.1 / 0.2 = 0.5

    split_temp2 = rest_ds.train_test_split(
        test_size=test_size_real, seed=random_seed, shuffle=True
    )

    val_ds = split_temp2['train']
    test_ds = split_temp2['test']

    print(f"Validation: {len(val_ds)}개 저장 중...")
    val_ds.save_to_disk(f"{save_dir}/validation")
    
    print(f"Test: {len(test_ds)}개 저장 중...")
    test_ds.save_to_disk(f"{save_dir}/test")

    # 캐시 삭제 (필요한 경우에만 주석 해제하여 사용)
    # shutil.rmtree("/root/.cache/huggingface/datasets", ignore_errors=True)

    print("\n✅ 전처리 완료: config_name 컬럼이 포함된 데이터셋이 저장되었습니다.")


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
        train_ratio=0.8,  # 비율 조정 (일반적으로 8:1:1 많이 사용)
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )