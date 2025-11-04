# ============================================================================
# 👤 PERSON D: 파일 2: README.md
# ============================================================================

"""
README.md 내용 (마크다운 형식):
"""

# English Dialects Accent Classification

## 🎯 프로젝트 목표

영국 6개 지역의 억양 분류 (기본 태스크: 지역 분류, 보조 태스크: 성별 분류)

**Novelty:**
1. **Attention-based Geographic Embedding** - 위도/경도 정보를 attention으로 음성 특징과 융합
2. **Partial Fine-tuning** - Wav2Vec2 상위 4개 레이어만 학습으로 효율성 극대화

## 📊 데이터셋

- **출처**: HuggingFace - `ylacombe/english_dialects`
- **지역** (6개): Irish, Midlands, Northern, Scottish, Southern, Welsh
- **성별** (2개): Male, Female
- **총 샘플**: ~14,000
- **총 시간**: ~31시간

## 🗂️ 프로젝트 구조

```
project/
├── data/
│   ├── data_config.py           # 레이블, 좌표 매핑
│   ├── dataset.py               # Custom Dataset
│   └── preprocessing.py         # 오디오 전처리, SpecAugment
├── models/
│   └── baseline.py              # Wav2Vec2 + Classification Head
├── train.py                     # 학습 스크립트
├── evaluate.py                  # 평가 스크립트
├── metrics.py                   # 평가 메트릭
├── visualize.py                 # 시각화 함수
├── configs/
│   └── experiment_config.yaml   # 하이퍼파라미터
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정 (Colab)
```bash
!pip install torch torchaudio transformers datasets scikit-learn matplotlib seaborn pyyaml
```

### 2. 학습
```python
from train import main
main()
```

### 3. 평가
```python
from evaluate import Evaluator
evaluator = Evaluator('checkpoints/best_model.pt', config)
results = evaluator.evaluate()
```

## 📈 실험 설정

### Baseline
- Wav2Vec2 (12 레이어 모두 학습)
- Simple pooling + linear classifier

### Ours (목표 모델)
- Wav2Vec2 (상위 4개 레이어만 학습)
- Geographic embedding + Attention fusion

### 하이퍼파라미터
- Learning rate: 5e-5
- Batch size: 8 (Colab GPU 권장)
- Epochs: 30
- Optimizer: AdamW

## 👥 팀 역할

- **Person A**: 데이터 파이프라인 (Dataset, DataLoader)
- **Person B**: 베이스라인 모델 (Wav2Vec2, 학습 루프)
- **Person C**: 평가 및 실험 관리 (Metrics, 결과 저장)
- **Person D**: 문서화 및 시각화

## 📝 체크리스트

- [ ] Week 1: 베이스라인 학습 시작
- [ ] Week 2: 모든 실험 완료
- [ ] Week 3: 분석 및 보고서 작성

## 🔗 참고 자료

- [Wav2Vec2 논문](https://arxiv.org/abs/2006.11477)
- [XLSR (다국어 Wav2Vec2)](https://arxiv.org/abs/2111.16268)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## 📧 문의

프로젝트 관련 질문은 리더에게 문의하세요.
