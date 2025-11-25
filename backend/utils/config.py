"""
utils/config.py
Configuration for GeoAccent British English Accent Classifier.
Defines hyperparameters, model settings, and file paths.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from pytz import timezone
from transformers import Wav2Vec2Config

# Dataset Labels and Geographic Coordinates
REGION_LABELS = {'irish': 0, 'midlands': 1, 'northern': 2, 'scottish': 3, 'southern': 4, 'welsh': 5}
GENDER_LABELS = {'male': 0, 'female': 1}
ID_TO_REGION = {v: k for k, v in REGION_LABELS.items()}
ID_TO_GENDER = {v: k for k, v in GENDER_LABELS.items()}
REGION_COORDS = {
    'irish': (53.3498, -6.2603), 'midlands': (52.6569, -1.1398), 'northern': (54.5973, -5.9301),
    'scottish': (55.9533, -3.1883), 'southern': (51.5074, -0.1278), 'welsh': (51.4816, -3.1791)
}
LAT_MIN, LAT_MAX = 51.4, 55.9
LON_MIN, LON_MAX = -6.3, -0.1

def normalize_coords(lat: float, lon: float) -> Tuple[float, float]:
    norm_lat = 2 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1
    norm_lon = 2 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1
    return norm_lat, norm_lon

def get_region_coordinates(region: str) -> Tuple[float, float]:
    lat, lon = REGION_COORDS[region.lower()]
    return normalize_coords(lat, lon)

@dataclass
class GeoAccentConfig:
    # Model Architecture
    pretrained_model: str = "facebook/wav2vec2-large-xlsr-53"
    num_regions: int = 6
    num_genders: int = 2
    hidden_dim: Optional[int] = None
    geo_embedding_dim: int = 256
    fusion_dim: int = 512
    dropout: float = 0.1
    use_fusion: bool = True

    # Fine-tuning
    freeze_lower_layers: bool = True
    num_frozen_layers: int = 16

    # Loss Weights
    region_weight: float = 1.0
    gender_weight: float = 0.1
    distance_weight: float = 0.05

    # Training
    batch_size: int = 8
    eval_batch_size: int = 12
    gradient_accumulation_steps: int = 2
    num_epochs: int = 25
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0

    # Optimization
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    scheduler_type: str = "cosine"
    warmup_steps: int = 500

    # Early Stopping
    early_stopping_patience: int = 5
    min_delta: float = 0.001

    # Data
    dataset_name: str = "ylacombe/english_dialects"
    audio_sample_rate: int = 16000
    max_audio_length: int = 30
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    device: str = "cuda"

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 50
    eval_steps: int = 500

    # Experiment Tracking
    use_wandb: bool = False
    wandb_project: str = "geo-accent-classifier"
    wandb_run_name: Optional[str] = None

    # Directories
    base_experiment_dir: str = "experiments"
    experiment_name: Optional[str] = None
    data_dir: str = "data"
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    result_dir: Optional[str] = None

    # Misc
    seed: int = 42

    def __post_init__(self):
        self._setup_model_dimensions()
        self._normalize_loss_weights()
        self._setup_experiment_paths()
        self._validate_config()

    def _setup_model_dimensions(self):
        if self.hidden_dim is not None:
            return
        try:
            wav2vec_config = Wav2Vec2Config.from_pretrained(self.pretrained_model)
            self.hidden_dim = wav2vec_config.hidden_size
        except Exception:
            self.hidden_dim = 1024

    def _normalize_loss_weights(self):
        total = self.region_weight + self.gender_weight + self.distance_weight
        if abs(total - 1.0) > 1e-6:
            self.region_weight /= total
            self.gender_weight /= total
            self.distance_weight /= total

    def _setup_experiment_paths(self):
        if self.experiment_name is None:
            timestamp = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
            self.experiment_name = "_".join([
                'geo_accent', self.pretrained_model.split('/')[-1],
                f'freeze{self.num_frozen_layers}',
                f'bs{self.batch_size}x{self.gradient_accumulation_steps}',
                timestamp
            ])
        experiment_root = os.path.join(self.base_experiment_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(experiment_root, 'checkpoints')
        self.log_dir = os.path.join(experiment_root, 'logs')
        self.result_dir = os.path.join(experiment_root, 'results')
        if self.wandb_run_name is None:
            self.wandb_run_name = self.experiment_name

    def _validate_config(self):
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio:.3f}")
        if self.freeze_lower_layers and (self.num_frozen_layers < 0 or self.num_frozen_layers > 24):
            raise ValueError(f"num_frozen_layers must be in [0, 24], got {self.num_frozen_layers}")
        if self.amp_dtype not in ['float16', 'bfloat16']:
            raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype}")

    def get_effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def get_trainable_params_ratio(self) -> float:
        if not self.freeze_lower_layers:
            return 1.0
        return (24 - self.num_frozen_layers) / 24

    def create_directories(self):
        for directory in [self.checkpoint_dir, self.log_dir, self.result_dir]:
            os.makedirs(directory, exist_ok=True)

    def save_config(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.base_experiment_dir, self.experiment_name, 'config.json')
        config_dict = {attr: getattr(self, attr) for attr in dir(self)
                       if not attr.startswith('_') and not callable(getattr(self, attr))}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_config(cls, path: str) -> 'GeoAccentConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
