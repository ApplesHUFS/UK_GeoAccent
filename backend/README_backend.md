# Backend - API 서버

GeoAccent 영국 억양 분류 시스템의 **Flask 기반 백엔드 API 서버**입니다.  
Frontend에서 업로드된 음성을 받아 모델 추론 또는 fallback 모드를 통헤 억양 예측 결과를 반환합니다.

---

## 주요 기능

- **음성 업로드 처리**: Frontend에서 전달된 오디오 파일 수신 및 기본 검증
- **GeoAccent 모델 추론**: Wav2Vec2 기반 음성 특성 + Geographic Embedding + Attention Fusion 적용
- **Partial Fine-tuning**: 하위 레이어 동결을 통해 학습 효율성과 안정성 향상
- **Fallback 모드 지원**: 모델 연동 전에도 API 및 Frontend 연동을 테스트할 수 있는 더미 추론 제공
- **CORS 지원**: 브라우저 환경에서의 Cross-Origin 요청 허용
- **JSON 기반 응답**: Frontend에서 바로 활용 가능한 일관된 응답 포맷 제공

---

## 기술 스택

- **Flask**: 웹 프레임워크
- **PyTorch**: Wav2Vec2 기반 딥러닝 모델 로드 및 추론
- **Transformers**: Wav2Vec2 모델 및 Config 로딩
- **HuggingFace Datasets**: `ylacombe/english_dialects` 데이터셋 로딩 및 가공
- **Librosa / SoundFile**: 오디오 파일 로딩 및 전처리
- **NumPy / Scikit-learn**: 메트릭 계산 및 평가
- **Matplotlib / Seaborn**: 학습 곡선 및 Confusion Matrix 시각화

---

## 아키텍처

백엔드의 전체 흐름은 다음과 같습니다.

1. **Frontend → Backend로 오디오 업로드**
   - 사용자가 브라우저에서 녹음 또는 파일 업로드
   - `multipart/form-data` 형식으로 `/api/classify` 호출

2. **`app.py`에서 파일 수신**
   - Flask가 `request.files["audio"]`로 파일을 수신
   - 파일 유효성(존재 여부, 파일명 등) 체크

3. **(모델 연동 시) 오디오 전처리**
   - 16kHz resampling 및 mono 변환
   - `AudioPreprocessor`를 통해 normalize
   - (훈련 시) `SpecAugment`를 통한 데이터 증강

4. **GeoAccentClassifier 추론**
   - Wav2Vec2를 이용한 음성 feature 추출
   - `GeoEmbedding`으로 지리 정보(lat, lon)를 embedding 벡터로 변환
   - `AttentionFusion`으로 음성 + 지리 정보를 동적으로 융합
   - 지역 분류(region) + 성별 분류(gender) + 지역 임베딩 예측 수행

5. **결과를 JSON으로 반환**
   - 최종 예측 지역 이름
   - 각 지역에 대한 확률 분포
   - 최종 confidence score


---

## 디렉토리 구조

```bash
backend/
├── app.py                      # Flask 메인 애플리케이션 (API 엔드포인트)
├── main.py                     # Preprocess / Train / Evaluate 통합 실행 엔트리
├── preprocess.py               # HF Dataset → train/val/test split
├── download_dataset.sh         # English Dialects 데이터 다운로드 스크립트
├── requirements.txt            # Python 의존성 목록
│
├── models/                     # 모델 아키텍처 구성 요소
│   ├── classifier.py           # GeoAccentClassifier (Wav2Vec2 + GeoEmbedding + Attention)
│   ├── embeddings.py           # GeoEmbedding, AttentionFusion 모듈
│   ├── losses.py               # MultiTaskLossWithDistance (region/gender/distance loss)
│   └── __init__.py
│
├── preprocessing/              # 오디오 전처리 및 SpecAugment
│   ├── preprocessing.py        # AudioPreprocessor, SpecAugment
│   └── __init__.py
│
├── train/                      # 훈련 루프 및 Trainer
│   ├── trainer.py              # AccentTrainer (학습/검증/체크포인트 관리)
│   ├── train.py                # 명령행 인자 파싱 + train_model 진입점
│   └── __init__.py
│
├── evaluation/                 # 모델 평가
│   ├── evaluate.py             # Evaluate pipeline + metrics/CM 저장
│   ├── metrics.py              # 정확도·F1·정밀도 등 평가 지표 계산 모듈
│   └── __init__.py
│
└── utils/
    ├── config.py               # GeoAccentConfig, REGION_LABELS, REGION_COORDS 등 전체 설정 및 지리 좌표 관리 
    └── visualization.py        # 학습 곡선·Per-class metrics·Waveform·Spectrogram 시각화

```

## API 엔드포인트

---

### 1. Health Check

**GET** `/api/health`

### 응답 예시

```json
{
  "status": "ok",
  "message": "Accent Classifier API is running"
}
```

---

### 2. 억양 분류 API

**POST** `/api/classify`

Form-data에 `audio` 필드를 포함해야 합니다.

- 필수 필드:
  - `audio`: 업로드할 음성 파일 (예: `recording.wav`)

### 응답 예시 (fallback 모드)

```json
{
  "accent": "Irish",
  "confidence": 0.62,
  "allProbabilities": {
    "Irish": 0.62,
    "Midland": 0.09,
    "Northern": 0.10,
    "Scottish": 0.06,
    "Southern": 0.07,
    "Welsh": 0.06
  },
  "mode": "fallback"
}
```

---

### 3. 에러 응답 예시

```json
{
  "error": "No audio file provided"
}
```

## 실행 방법 
### 1. 가상환경 생성
### 2. 패키지 설치
### 3. 서버 실행 
## 데이터셋 다운로드
## 학습 실행 
## 평가 실행 

