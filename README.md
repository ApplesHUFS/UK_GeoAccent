
# English Dialects Accent Classification
---
## 🎯 프로젝트 목표

UK_GeoAccent는 영국 영어 억양을 분류하고, 음성 샘플로부터 지리적 위치를 예측하는 모델입니다. Wav2Vec2 기반 사전학습 모델을 활용하여 6가지 주요 영국 지역의 억양을 인식하고, 각 지역의 정규화된 좌표를 예측하는 것을 목표로 합니다.

**Novelty:**
1. **Attention-based Geographic Embedding** - 위도/경도 정보를 attention으로 음성 특징과 융합
2. **Partial Fine-tuning** - Wav2Vec2 상위 8개 레이어만 학습하여 효율성 극대화

## 📊 데이터셋
이 프로젝트는 HuggingFace의 [ylacombe/english_dialects](https://huggingface.co/datasets/ylacombe/english_dialects?library=datasets) 데이터셋을 사용합니다.
- **지역** (6개): Irish, Midlands, Northern, Scottish, Southern, Welsh
    - 지역 좌표 정보
    |지역|좌표|도시|
    |--------|----------------|----------|
    |Irish|53.3498, -6.2603|Dublin|
    |Midlands|52.6569, -1.1398|Birmingham|
    |Northern|54.5973, -5.9301|Belfast|
    |Scottish|55.9533, -3.1883|Edinburgh|
    |Southern|51.5074, -0.1278|London|
    |Welsh|51.4816, -3.1791|Cardiff|
- **성별** (2개): Male, Female
- **sampling rate**: 16,000Hz
- **총 시간**: ~31시간

## 🗂️ 프로젝트 구조

```
UK_GeoAccent/
├── configs/
│   └── experiment_config.yaml   # 하이퍼파라미터
├── data/
│   ├── __init__.py   
│   ├── data_config.py           # 레이블, 좌표 매핑
│   ├── dataset.py               # Custom Dataset
│   └── preprocessing.py         # 오디오 전처리 및 SpecAugment
├── models/
│   ├── __init__.py
│   └── baseline.py              # Wav2Vec2 + Classification Head
├── utils
│   ├── .gitignore   
│   ├── README.md
│   ├── evaluate.py              # 평가 스크립트
│   ├── requirement.txt 
│   └── train.py                 # 학습 스크립트
```

## 🚀 빠른 시작

### 1. 환경 설정
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
- Wav2Vec2 (24 레이어 모두 학습)
- Simple pooling + linear classifier

### Ours (목표 모델)
- Wav2Vec2 (상위 8개 레이어만 학습)
- Geographic embedding + Attention fusion

### 하이퍼파라미터
- Learning rate: 5e-5
- Batch size: 8 (Colab GPU 권장)
- Epochs: 30
- Optimizer: AdamW


## 🔗 참고 자료

- [Wav2Vec2 논문](https://arxiv.org/abs/2006.11477)
- [XLSR (다국어 Wav2Vec2)](https://arxiv.org/abs/2111.16268)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## 📧 문의

프로젝트 관련 질문은 리더에게 문의하세요.
