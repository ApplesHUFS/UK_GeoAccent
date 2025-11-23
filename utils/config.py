"""Configuration for GeoAccent British English Accent Classifier.

This module defines all hyperparameters, model settings, and file paths for
the geographic-aware accent classification system with automatic model 
architecture adaptation and RTX 4090 24GB optimization.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pytz import timezone
from transformers import Wav2Vec2Config


# ============================================================================
# Constants: Dataset Labels and Geographic Coordinates
# ============================================================================

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

# Reverse mappings
ID_TO_REGION = {v: k for k, v in REGION_LABELS.items()}
ID_TO_GENDER = {v: k for k, v in GENDER_LABELS.items()}

# Geographic coordinates (lat, lon)
REGION_COORDS = {
    'irish': (53.3498, -6.2603),      # Dublin
    'midlands': (52.6569, -1.1398),   # Birmingham
    'northern': (54.5973, -5.9301),   # Belfast
    'scottish': (55.9533, -3.1883),   # Edinburgh
    'southern': (51.5074, -0.1278),   # London
    'welsh': (51.4816, -3.1791)       # Cardiff
}

# Coordinate normalization ranges
LAT_MIN, LAT_MAX = 51.4, 55.9
LON_MIN, LON_MAX = -6.3, -0.1


def normalize_coords(lat: float, lon: float) -> Tuple[float, float]:
    """Normalize latitude/longitude to [-1, 1] range.
    
    Args:
        lat: Latitude value
        lon: Longitude value
    
    Returns:
        Tuple of (normalized_lat, normalized_lon) in [-1, 1]
    """
    norm_lat = 2 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1
    norm_lon = 2 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1
    return norm_lat, norm_lon


def get_region_coordinates(region: str) -> Tuple[float, float]:
    """Get normalized coordinates for a region.
    
    Args:
        region: Region name (e.g., 'irish', 'scottish')
    
    Returns:
        Tuple of normalized (lat, lon) coordinates
    """
    lat, lon = REGION_COORDS[region.lower()]
    return normalize_coords(lat, lon)


# ============================================================================
# Main Configuration Class
# ============================================================================

@dataclass
class GeoAccentConfig:
    """Configuration for GeoAccent classifier training and evaluation.
    
    This class contains all hyperparameters and settings optimized for
    RTX 4090 24GB GPU training. It automatically adapts to the pretrained
    Wav2Vec2 model architecture and handles all path setup.
    
    Attributes:
        # Model Architecture
        pretrained_model: Hugging Face model identifier
        num_regions: Number of region classes
        num_genders: Number of gender classes
        hidden_dim: Wav2Vec2 hidden dimension (auto-detected)
        geo_embedding_dim: Geographic embedding dimension
        fusion_dim: Attention fusion dimension
        dropout: Dropout rate for regularization
        
        # Fine-tuning Strategy
        freeze_lower_layers: Whether to freeze lower Wav2Vec2 layers
        num_frozen_layers: Number of layers to freeze (out of 24)
        
        # Loss Configuration
        region_weight: Weight for region classification loss
        gender_weight: Weight for gender classification loss (auxiliary)
        distance_weight: Weight for distance regularization loss
        
        # Training Configuration
        batch_size: Training batch size (RTX 4090 optimized)
        eval_batch_size: Evaluation batch size
        gradient_accumulation_steps: Steps to accumulate gradients
        num_epochs: Total training epochs
        learning_rate: Learning rate for trainable parameters
        max_grad_norm: Gradient clipping threshold
        
        # Optimization
        use_amp: Enable Automatic Mixed Precision
        amp_dtype: AMP dtype ('float16' or 'bfloat16')
        scheduler_type: LR scheduler ('cosine' or 'linear')
        warmup_steps: Warmup steps for scheduler
        
        # Early Stopping
        early_stopping_patience: Epochs to wait before stopping
        min_delta: Minimum improvement to count as progress
        
        # Data Configuration
        dataset_name: HuggingFace dataset identifier
        audio_sample_rate: Audio sampling rate in Hz
        max_audio_length: Maximum audio length in seconds
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
        # Hardware Configuration
        num_workers: Number of DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        device: Training device ('cuda' or 'cpu')
        
        # Checkpointing
        save_strategy: When to save checkpoints ('epoch' or 'steps')
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        logging_steps: Log training metrics every N steps
        eval_steps: Run evaluation every N steps
        
        # Experiment Tracking
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        
        # Directory Paths
        base_experiment_dir: Root directory for experiments
        experiment_name: Name of current experiment (auto-generated)
        data_dir: Root directory containing dataset
        checkpoint_dir: Checkpoint save directory
        log_dir: Log save directory
        result_dir: Results save directory
        
        # Misc
        seed: Random seed for reproducibility
    """
    
    # ========================================================================
    # Model Architecture
    # ========================================================================
    pretrained_model: str = "facebook/wav2vec2-large-xlsr-53"
    num_regions: int = 6
    num_genders: int = 2
    hidden_dim: Optional[int] = None  # Auto-detected from pretrained model
    geo_embedding_dim: int = 256
    fusion_dim: int = 512
    dropout: float = 0.1
    
    # ========================================================================
    # Fine-tuning Strategy
    # ========================================================================
    freeze_lower_layers: bool = True
    num_frozen_layers: int = 16  # Freeze lower 16 out of 24 layers
    
    # ========================================================================
    # Loss Configuration
    # ========================================================================
    region_weight: float = 1.0
    gender_weight: float = 0.3
    distance_weight: float = 0.5
    
    # ========================================================================
    # Training Configuration (RTX 4090 24GB Optimized)
    # ========================================================================
    batch_size: int = 4  # Conservative for stability
    eval_batch_size: int = 8  # Can be larger during eval
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    num_epochs: int = 25
    learning_rate: float = 1e-5  # Low LR for partial fine-tuning
    max_grad_norm: float = 1.0
    
    # ========================================================================
    # Optimization
    # ========================================================================
    use_amp: bool = True  # 30% speedup with mixed precision
    amp_dtype: str = "bfloat16"  # RTX 4090 supports bf16 (more stable than fp16)
    scheduler_type: str = "cosine"  # 'cosine' or 'linear'
    warmup_steps: int = 500
    
    # ========================================================================
    # Early Stopping
    # ========================================================================
    early_stopping_patience: int = 5
    min_delta: float = 0.001
    
    # ========================================================================
    # Data Configuration
    # ========================================================================
    dataset_name: str = "ylacombe/english_dialects"
    audio_sample_rate: int = 16000  # Wav2Vec2 default
    max_audio_length: int = 30  # seconds
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # ========================================================================
    # Hardware Configuration
    # ========================================================================
    num_workers: int = 4  # Out of 6 vCPUs
    pin_memory: bool = True  # Faster GPU transfer
    persistent_workers: bool = True  # Reduce overhead
    device: str = "cuda"
    
    # ========================================================================
    # Checkpointing
    # ========================================================================
    save_strategy: str = "steps"  # 'epoch' or 'steps'
    save_steps: int = 500  # For Spot Instance resilience
    save_total_limit: int = 3  # Keep only 3 most recent checkpoints
    logging_steps: int = 50
    eval_steps: int = 500
    
    # ========================================================================
    # Experiment Tracking
    # ========================================================================
    use_wandb: bool = False
    wandb_project: str = "geo-accent-classifier"
    wandb_run_name: Optional[str] = None  # Auto-generated
    
    # ========================================================================
    # Directory Paths
    # ========================================================================
    base_experiment_dir: str = "experiments"
    experiment_name: Optional[str] = None  # Auto-generated with timestamp
    data_dir: str = "data"
    checkpoint_dir: Optional[str] = None  # Set in __post_init__
    log_dir: Optional[str] = None  # Set in __post_init__
    result_dir: Optional[str] = None  # Set in __post_init__
    
    # ========================================================================
    # Misc
    # ========================================================================
    seed: int = 42
    
    def __post_init__(self):
        """Initialize configuration after dataclass initialization.
        
        This method:
          1. Auto-detects hidden dimension from pretrained model
          2. Normalizes loss weights to sum to 1.0
          3. Sets up experiment directory structure
          4. Validates configuration parameters
        """
        self._setup_model_dimensions()
        self._normalize_loss_weights()
        self._setup_experiment_paths()
        self._validate_config()
    
    def _setup_model_dimensions(self):
        """Auto-detect hidden dimension from pretrained Wav2Vec2 model.
        
        Loads the pretrained model config and extracts hidden_size.
        Falls back to 1024 if loading fails.
        """
        if self.hidden_dim is not None:
            return  # Already set manually
        
        try:
            wav2vec_config = Wav2Vec2Config.from_pretrained(self.pretrained_model)
            self.hidden_dim = wav2vec_config.hidden_size
            print(f"✅ Detected hidden_dim: {self.hidden_dim} from {self.pretrained_model}")
        except Exception as e:
            print(f"⚠️  Could not load Wav2Vec2 config: {e}")
            self.hidden_dim = 1024
            print(f"Using default hidden_dim: {self.hidden_dim}")
    
    def _normalize_loss_weights(self):
        """Normalize loss weights to sum to 1.0.
        
        Ensures that region_weight + gender_weight + distance_weight = 1.0
        for stable multi-task learning.
        """
        total = self.region_weight + self.gender_weight + self.distance_weight
        
        if abs(total - 1.0) > 1e-6:
            print(f"⚠️  Loss weights sum to {total:.3f}, normalizing to 1.0")
            self.region_weight /= total
            self.gender_weight /= total
            self.distance_weight /= total
            
        print(f"📊 Loss weights: Region={self.region_weight:.2f}, "
              f"Gender={self.gender_weight:.2f}, "
              f"Distance={self.distance_weight:.2f}")
    
    def _setup_experiment_paths(self):
        """Set up experiment directory structure.
        
        Creates a unique experiment name with timestamp if not provided.
        Configures checkpoint, log, and result directories.
        """
        if self.experiment_name is None:
            timestamp = datetime.now(
                timezone('Asia/Seoul')
            ).strftime('%Y%m%d_%H%M%S')
            
            # Generate experiment name
            name_parts = [
                'geo_accent',
                self.pretrained_model.split('/')[-1],  # e.g., 'wav2vec2-large-xlsr-53'
                f'freeze{self.num_frozen_layers}',
                f'bs{self.batch_size}x{self.gradient_accumulation_steps}',
                timestamp
            ]
            
            self.experiment_name = "_".join(name_parts)
        
        # Set up directory paths
        experiment_root = os.path.join(
            self.base_experiment_dir,
            self.experiment_name
        )
        
        self.checkpoint_dir = os.path.join(experiment_root, 'checkpoints')
        self.log_dir = os.path.join(experiment_root, 'logs')
        self.result_dir = os.path.join(experiment_root, 'results')
        
        # Set W&B run name
        if self.wandb_run_name is None:
            self.wandb_run_name = self.experiment_name
    
    def _validate_config(self):
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check split ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Data split ratios must sum to 1.0, got {total_ratio:.3f}"
            )
        
        # Check frozen layers
        if self.freeze_lower_layers:
            if self.num_frozen_layers < 0 or self.num_frozen_layers > 24:
                raise ValueError(
                    f"num_frozen_layers must be in [0, 24], got {self.num_frozen_layers}"
                )
        
        # Check batch configuration
        if self.batch_size * self.gradient_accumulation_steps < 8:
            print(f"⚠️  Effective batch size is very small: "
                  f"{self.batch_size * self.gradient_accumulation_steps}")
        
        # Check AMP dtype
        if self.amp_dtype not in ['float16', 'bfloat16']:
            raise ValueError(
                f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype}"
            )
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size after gradient accumulation.
        
        Returns:
            Effective batch size (batch_size × gradient_accumulation_steps)
        """
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_trainable_params_ratio(self) -> float:
        """Get ratio of trainable parameters (approximate).
        
        Returns:
            Approximate ratio of trainable parameters
        """
        if not self.freeze_lower_layers:
            return 1.0
        
        # XLSR-53 has 24 layers
        return (24 - self.num_frozen_layers) / 24
    
    def create_directories(self):
        """Create all necessary directories for the experiment."""
        for directory in [self.checkpoint_dir, self.log_dir, self.result_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def save_config(self, path: Optional[str] = None):
        """Save configuration to JSON file.
        
        Args:
            path: Output file path. If None, saves to experiment_dir/config.json
        """
        if path is None:
            path = os.path.join(self.base_experiment_dir, self.experiment_name, 'config.json')
        
        # Convert config to dict
        config_dict = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        
        # Save to file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Config saved to: {path}")
    
    @classmethod
    def load_config(cls, path: str) -> 'GeoAccentConfig':
        """Load configuration from JSON file.
        
        Args:
            path: Path to config JSON file
        
        Returns:
            GeoAccentConfig instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*80)
        print("GeoAccent Configuration Summary")
        print("="*80)
        
        print("\n🎯 Model Architecture:")
        print(f"  Pretrained: {self.pretrained_model}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Geo embedding dim: {self.geo_embedding_dim}")
        print(f"  Fusion dim: {self.fusion_dim}")
        print(f"  Frozen layers: {self.num_frozen_layers}/24 "
              f"({self.get_trainable_params_ratio()*100:.0f}% trainable)")
        
        print("\n📊 Training Configuration:")
        print(f"  Batch size: {self.batch_size} × {self.gradient_accumulation_steps} "
              f"= {self.get_effective_batch_size()} (effective)")
        print(f"  Learning rate: {self.learning_rate:.2e}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  AMP: {self.use_amp} ({self.amp_dtype})")
        print(f"  Scheduler: {self.scheduler_type} (warmup={self.warmup_steps})")
        
        print("\n⚖️  Loss Weights:")
        print(f"  Region: {self.region_weight:.2f}")
        print(f"  Gender: {self.gender_weight:.2f}")
        print(f"  Distance: {self.distance_weight:.2f}")
        
        print("\n💾 Checkpointing:")
        print(f"  Strategy: {self.save_strategy}")
        print(f"  Save every: {self.save_steps} steps")
        print(f"  Keep: {self.save_total_limit} checkpoints")
        
        print("\n📁 Directories:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        
        if self.use_wandb:
            print("\n📈 Weights & Biases:")
            print(f"  Project: {self.wandb_project}")
            print(f"  Run: {self.wandb_run_name}")
        
        print("\n" + "="*80 + "\n")


# ============================================================================
# Predefined Configurations
# ============================================================================

def get_default_config() -> GeoAccentConfig:
    """Get default configuration for RTX 4090 24GB.
    
    Returns:
        Default GeoAccentConfig instance
    """
    return GeoAccentConfig()


def get_quick_test_config() -> GeoAccentConfig:
    """Get configuration for quick testing (1 epoch, small batch).
    
    Returns:
        GeoAccentConfig for quick testing
    """
    return GeoAccentConfig(
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        experiment_name="quick_test"
    )


def get_full_training_config() -> GeoAccentConfig:
    """Get configuration for full training run.
    
    Returns:
        GeoAccentConfig optimized for full 31h dataset
    """
    return GeoAccentConfig(
        num_epochs=30,
        batch_size=8,  # Higher batch for full training
        gradient_accumulation_steps=2,
        use_wandb=True,
        early_stopping_patience=7
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create default config
    config = get_default_config()
    
    # Print summary
    config.print_summary()
    
    # Create directories
    config.create_directories()
    
    # Save config
    config.save_config()
    
    # Example: Get region coordinates
    print("\n🗺️  Region Coordinates:")
    for region in REGION_LABELS.keys():
        lat, lon = get_region_coordinates(region)
        print(f"  {region:10s}: ({lat:.3f}, {lon:.3f})")