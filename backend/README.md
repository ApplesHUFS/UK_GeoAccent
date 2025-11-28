# GeoAccent: Geographic-Aware British English Accent Classifier

A Wav2Vec2-based deep learning model for British English accent classification with geographic information integration. The model combines audio features and geographic coordinates through attention mechanism for improved regional classification.

## Features

- **Geographic Attention Fusion**: Dynamic integration of audio and geographic information
- **Partial Fine-tuning**: Efficient training by freezing lower layers
- **Distance Regularization**: Explicit learning of geographic structure
- **Multi-task Learning**: Region classification + gender classification (auxiliary task)
- **Memory-efficient Pipeline**: Streaming data processing with JSON metadata

## Supported Regions

| Region | City | Coordinates |
|--------|------|-------------|
| Irish | Dublin | 53.3°N, 6.3°W |
| Midlands | Birmingham | 52.7°N, 1.1°W |
| Northern | Belfast | 54.6°N, 5.9°W |
| Scottish | Edinburgh | 56.0°N, 3.2°W |
| Southern | London | 51.5°N, 0.1°W |
| Welsh | Cardiff | 51.5°N, 3.2°W |

## Requirements

### Hardware
- **GPU**: CUDA-enabled GPU recommended (minimum 8GB VRAM, RTX 4090 24GB optimal)
- **RAM**: Minimum 16GB
- **Disk**: 30GB+ (dataset + model)

### Software
- Python 3.8+
- CUDA 11.0+ (for GPU usage)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/GeoAccent.git
cd GeoAccent/backend
```

### 2. Create Virtual Environment
```bash
# Conda (recommended)
conda create -n geoaccent python=3.10
conda activate geoaccent

# Or venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Download Dataset

First, download the English Dialects dataset from HuggingFace:

```bash
bash download_dataset.sh
```

This script downloads the Parquet files for all 11 configurations:
- irish_male
- midlands_female, midlands_male
- northern_female, northern_male
- scottish_female, scottish_male
- southern_female, southern_male
- welsh_female, welsh_male

The dataset will be saved to `../data/english_dialects/`.

### Step 2: Prepare Dataset

Convert the Parquet files to WAV+JSON format and split into train/validation/test:

```bash
python main.py prepare \
    --parquet_dir ../data/english_dialects \
    --save_dir ./data/english_dialects \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

This command:
1. Reads Parquet files with speaker metadata
2. Splits data by speaker (no speaker overlap between splits)
3. Converts audio to WAV files
4. Generates JSON metadata with labels and coordinates
5. Creates train/validation/test directories

Expected output structure:
```
data/english_dialects/
├── train/
│   ├── audio/
│   │   ├── 0.wav
│   │   ├── 1.wav
│   │   └── ...
│   └── metadata.json
├── validation/
│   ├── audio/
│   └── metadata.json
└── test/
    ├── audio/
    └── metadata.json
```

### Step 3: Train Model

**Basic training**:
```bash
python main.py train
```

**Custom configuration**:
```bash
python main.py train \
    --batch_size 8 \
    --num_epochs 40 \
    --learning_rate 1e-5 \
    --use_fusion \
    --num_frozen_layers 0
```

### Step 4: Evaluate Model

```bash
python main.py evaluate \
    --checkpoint checkpoints/best.pt \
    --split test \
    --output_dir results
```

## Training Configuration

### Model Architecture

**Use Geographic Fusion** (recommended):
```bash
python main.py train --use_fusion
```

**Without Fusion** (audio features only):
```bash
python main.py train
```

### Fine-tuning Strategy

**Full Fine-tuning** (all layers trainable):
```bash
python main.py train --num_frozen_layers 0
```

**Partial Fine-tuning** (freeze lower 16 layers):
```bash
python main.py train --num_frozen_layers 16
```

### Loss Weights

Adjust multi-task learning weights:
```bash
python main.py train \
    --region_weight 1.0 \
    --gender_weight 0.1 \
    --distance_weight 0.05
```

Loss composition:
```
Total Loss = α·L_region + β·L_gender + γ·L_distance

L_region:   Cross-Entropy (region classification, main task)
L_gender:   Cross-Entropy (gender classification, auxiliary)
L_distance: Cosine Distance (geographic embedding distance)
```

### Hyperparameters

Key hyperparameters:
```bash
python main.py train \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_epochs 40 \
    --warmup_steps 500 \
    --early_stopping_patience 8 \
    --max_grad_norm 1.0
```

### Optimization

**Mixed Precision Training** (enabled by default):
```bash
python main.py train --use_amp
```

**Resume Training**:
```bash
python main.py train --resume checkpoints/last.pt
```

## Project Structure

```
backend/
├── main.py                      # Main entry point
├── data/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset for WAV+JSON
│   └── prepare_dataset.py       # Parquet to WAV+JSON conversion
├── models/
│   ├── __init__.py
│   ├── classifier.py            # GeoAccentClassifier
│   ├── embeddings.py            # GeoEmbedding, AttentionFusion
│   └── losses.py                # MultiTaskLossWithDistance
├── train/
│   ├── __init__.py
│   ├── train.py                 # Training script
│   └── trainer.py               # AccentTrainer class
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py              # Evaluation script
│   └── metrics.py               # Evaluation metrics
├── preprocessing/
│   ├── __init__.py
│   └── preprocessing.py         # Audio preprocessing
├── utils/
│   └── config.py                # Configuration and constants
└── download_dataset.sh          # Dataset download script
```

## Evaluation Metrics

### Region Classification
- **Accuracy**: Overall accuracy
- **F1 Score (Macro)**: Class-balanced F1
- **F1 Score (Weighted)**: Sample-weighted F1
- **Precision**: Prediction precision
- **Recall**: Recall rate
- **Per-class Accuracy**: Accuracy for each region
- **Confusion Matrix**: Regional confusion patterns

### Gender Classification (Auxiliary)
- **Accuracy**: Gender classification accuracy
- **F1 Score**: Binary classification F1

### Geographic Embedding (with --use_fusion)
- **Cosine Similarity**: Predicted vs. actual embedding similarity
- **Distance Loss**: Geographic distance-based loss

## Troubleshooting

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution 1** - Reduce batch size:
```bash
python main.py train --batch_size 4 --gradient_accumulation_steps 4
```

**Solution 2** - Disable pin_memory (already done by default):
The code automatically sets `num_workers=0` and `pin_memory=False` for memory efficiency.

### DataLoader Worker Killed

**Symptom**: `RuntimeError: DataLoader worker (pid) is killed by signal: Aborted`

**Solution**: The code automatically uses `num_workers=0` to avoid multiprocessing memory issues. If you still encounter this error, reduce batch size:
```bash
python main.py train --batch_size 2
```

### Dataset Not Found

**Symptom**: `FileNotFoundError: Metadata not found`

**Solution**: Ensure you ran the prepare command:
```bash
python main.py prepare --parquet_dir ../data/english_dialects
```

### Training Instability

**Symptom**: Loss diverges or NaN values

**Solution 1** - Lower learning rate:
```bash
python main.py train --learning_rate 5e-6
```

**Solution 2** - Stronger gradient clipping:
```bash
python main.py train --max_grad_norm 0.5
```

**Solution 3** - Adjust loss weights:
```bash
python main.py train \
    --region_weight 1.0 \
    --gender_weight 0.05 \
    --distance_weight 0.01
```

## Performance Benchmark

### Training Time (RTX 4090 24GB, batch_size=8)

| Configuration | Time per Epoch | Full Training (40 epochs) |
|---------------|----------------|---------------------------|
| Full Fine-tuning | ~45 min | ~30 hours |
| Partial (16 frozen) | ~25 min | ~16 hours |

### Memory Usage

| Configuration | GPU Memory | Trainable Parameters |
|---------------|------------|---------------------|
| Full Fine-tuning | 18-20 GB | ~317M |
| Partial (16 frozen) | 12-14 GB | ~105M |
| No Fusion | 10-12 GB | Variable |

## Advanced Usage

### Python API

```python
from data.dataset import EnglishDialectsDataset, collate_fn
from models.classifier import GeoAccentClassifier
from train.trainer import AccentTrainer
from utils.config import REGION_LABELS

# Load dataset
train_dataset = EnglishDialectsDataset(
    split='train',
    data_dir='./data/english_dialects',
    audio_sample_rate=16000
)

# Create model
model = GeoAccentClassifier(
    model_name='facebook/wav2vec2-large-xlsr-53',
    num_regions=len(REGION_LABELS),
    num_genders=2,
    hidden_dim=1024,
    geo_embedding_dim=256,
    fusion_dim=512,
    dropout=0.1,
    freeze_lower_layers=True,
    num_frozen_layers=16,
    use_fusion=True
)

# Create trainer
trainer = AccentTrainer(
    model=model,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    learning_rate=1e-5,
    num_epochs=40
)

# Start training
trainer.train()
```

### Custom Data Preparation

If you have custom Parquet files:

```python
from data.prepare_dataset import convert_parquet_to_wav

convert_parquet_to_wav(
    parquet_dir='/path/to/parquet/files',
    save_dir='./data/custom_dataset',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)
```

## Data Format

### Metadata JSON Structure

Each split contains a `metadata.json` file with entries:

```json
[
  {
    "audio_path": "audio/0.wav",
    "text": "Transcript text",
    "speaker_id": "speaker_001",
    "config_name": "irish_male",
    "line_id": "line_001",
    "region_id": 0,
    "gender_id": 0,
    "normalized_lat": 0.814,
    "normalized_lon": -0.995
  },
  ...
]
```

### Coordinate Normalization

Coordinates are normalized to [-1, 1] range:
```python
normalized_lat = 2 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1
normalized_lon = 2 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1
```

Bounds:
- LAT_MIN: 51.4, LAT_MAX: 55.9
- LON_MIN: -6.3, LON_MAX: -0.1

## Citation

If you use this project, please cite:

```bibtex
@misc{geoaccent2024,
  title={GeoAccent: Geographic-Aware British English Accent Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GeoAccent}
}
```

## References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477) - Baevski et al., NeurIPS 2020
- [XLSR-53](https://arxiv.org/abs/2006.13979) - Conneau et al., Interspeech 2020
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., NeurIPS 2017
- [English Dialects Dataset](https://huggingface.co/datasets/ylacombe/english_dialects)

## License

MIT License - Free to use, modify, and distribute.

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Create a Pull Request

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/GeoAccent/issues)
- **Email**: your.email@example.com

## Acknowledgments

- [Hugging Face](https://huggingface.co) - Transformers library and datasets
- [Meta AI](https://www.meta.com/ai/) - Wav2Vec2 model
- [ylacombe](https://huggingface.co/ylacombe) - English Dialects Dataset
