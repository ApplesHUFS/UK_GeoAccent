import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import GeoAccentConfig
import json 

CONFIG_DIR = 'configs'
os.makedirs(CONFIG_DIR, exist_ok=True)
print(f"✅ Configs directory created/exists at: {CONFIG_DIR}")

# ----------------------------------------------------
# 1. 최종 모델 (Final Model) 설정
# ----------------------------------------------------
final_config = GeoAccentConfig(
    experiment_name='Final_Model_0_9357_Optimal' 
)
final_config_path = os.path.join(CONFIG_DIR, 'final_model.json')
final_config.save_config(final_config_path)
print(f"✅ Final Model config saved: {final_config_path}")

# ----------------------------------------------------
# 2. Baseline 1: Frozen XLSR-53 (전체 동결) 설정
# ----------------------------------------------------
baseline_config = GeoAccentConfig(
    num_frozen_layers=24,           # XLSR-53 전체 레이어 동결 (24 layers)
    distance_weight=0.0,            # Distance Loss 제거
    use_fusion=False,               # Geo-Fusion 비활성화
    experiment_name='baseline_frozen_xlsr'
)
baseline_config_path = os.path.join(CONFIG_DIR, 'baseline_frozen.json')
baseline_config.save_config(baseline_config_path)
print(f"✅ Baseline config saved: {baseline_config_path}")


# ----------------------------------------------------
# 3. Ablation 1: No Distance Loss (Geo-Fusion 유지) 설정
# ----------------------------------------------------
ablation_no_dist_config = GeoAccentConfig(
    # num_frozen_layers는 기본값 16을 유지합니다.
    distance_weight=0.0,            # Distance Loss만 제거
    use_fusion=True,                # Geo-Fusion 유지
    experiment_name='ablation_no_distance_loss'
)
ablation_no_dist_config_path = os.path.join(CONFIG_DIR, 'ablation_no_dist.json')
ablation_no_dist_config.save_config(ablation_no_dist_config_path)
print(f"✅ Ablation (No Dist Loss) config saved: {ablation_no_dist_config_path}")

# ----------------------------------------------------
# 4. Ablation 2: No Geo-Fusion (융합 로직 비활성화) 설정
# ----------------------------------------------------
ablation_no_geo_config = GeoAccentConfig(
    # num_frozen_layers는 기본값 16을 유지합니다.
    distance_weight=0.0,            # Distance Loss 제거
    use_fusion=False,               # <--- 이 부분이 Ablation 2의 핵심입니다.
    experiment_name='ablation_no_geo_fusion'
)
ablation_no_geo_config_path = os.path.join(CONFIG_DIR, 'ablation_no_geo.json')
ablation_no_geo_config.save_config(ablation_no_geo_config_path)
print(f"✅ Ablation (No Geo Fusion) config saved: {ablation_no_geo_config_path}")