"""
데이터셋 설정 및 상수 정의
- 지역 정보 (레이블 매핑)
- 위도/경도 좌표
- 데이터 경로 설정
"""

# ======================== 지역 레이블 매핑 ========================
REGION_LABELS = {
    'irish': 0,
    'midlands': 1,
    'northern': 2,
    'scottish': 3,
    'southern': 4,
    'welsh': 5
}

GENDER_LABELS = {
    'male': 0,
    'female': 1
}

# 역매핑
ID_TO_REGION = {v: k for k, v in REGION_LABELS.items()}
ID_TO_GENDER = {v: k for k, v in GENDER_LABELS.items()}

# ======================== 지리 좌표 ========================
REGION_COORDS = {
    'irish': (53.3498, -6.2603),      # Dublin
    'midlands': (52.6569, -1.1398),   # Birmingham
    'northern': (54.5973, -5.9301),   # Belfast
    'scottish': (55.9533, -3.1883),   # Edinburgh
    'southern': (51.5074, -0.1278),   # London
    'welsh': (51.4816, -3.1791)       # Cardiff
}

# 좌표 정규화를 위한 범위
LAT_MIN, LAT_MAX = 51.4, 55.9
LON_MIN, LON_MAX = -6.2, -0.1

def normalize_coords(lat, lon):
    """위도/경도 정규화 [-1, 1]"""
    norm_lat = 2 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1
    norm_lon = 2 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1
    return norm_lat, norm_lon

# ======================== 데이터셋 설정 ========================
DATASET_NAME = "ylacombe/english_dialects"
AUDIO_SAMPLE_RATE = 16000  # Wav2Vec2 기본 샘플링 레이트
MAX_AUDIO_LENGTH = 30  # 초 단위, 필요시 조정

# Train/Val/Test split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1