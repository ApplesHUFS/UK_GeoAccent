"""
data/dataset.py
Custom PyTorch Dataset
"""

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from datasets import load_from_disk

from utils.config import (
    REGION_LABELS, 
    GENDER_LABELS,
    REGION_COORDS, 
    normalize_coords
)
from preprocessing.preprocessing import AudioPreprocessor


class EnglishDialectsDataset(Dataset):
    """
    Loads a HuggingFace dataset and returns raw waveforms for Wav2Vec2.
    """
    
    def __init__(self, split='train', use_augment=False, 
                 data_dir="./data/english_dialects", audio_sample_rate=16000):
        super().__init__()
        self.split = split
        self.use_augment = use_augment
        self.audio_sample_rate = audio_sample_rate
        
        local_path = os.path.join(data_dir, split)
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Dataset not found at {local_path}."
            )
        
        try:
            self.dataset = load_from_disk(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {local_path}: {e}")
        
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.audio_sample_rate,
            use_augment=use_augment
        )
    
    def _parse_label(self, config_name):
        """
        Parse label: 'irish_male' → (region_id, gender_id, coords)
        """
        try:
            parts = config_name.split('_')
            region = parts[0]
            gender = parts[1]
            
            region_id = REGION_LABELS.get(region, 0)
            gender_id = GENDER_LABELS.get(gender, 0)
            
            lat, lon = REGION_COORDS.get(region, (0.0, 0.0))
            n_lat, n_lon = normalize_coords(lat, lon)
            coords = (n_lat, n_lon)
            
            return region_id, gender_id, coords
            
        except Exception:
            return 0, 0, (0.0, 0.0)

    def _process_waveform(self, audio_data, sr):
        """
        Resample and normalize audio waveform.
        """
        if sr != self.audio_sample_rate:
            audio_data = librosa.resample(
                y=np.array(audio_data, dtype=np.float32), 
                orig_sr=sr, 
                target_sr=self.audio_sample_rate
            )
        
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        waveform = self.preprocessor.normalize_audio(waveform)
        
        return waveform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        audio_info = sample["audio"]
        raw_audio = audio_info["array"]
        orig_sr = audio_info["sampling_rate"]
        
        waveform = self._process_waveform(raw_audio, orig_sr)
        
        config_name = sample["config_name"]
        region_id, gender_id, coords = self._parse_label(config_name)
        
        return {
            "waveform": waveform,
            "region_id": region_id,
            "gender_id": gender_id,
            "coords": coords,
            "config_name": config_name
        }


def collate_fn(batch):
    """
    Padding for variable-length audio and label stacking.
    """
    waveforms = [b["waveform"] for b in batch]
    
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    batch_size = len(batch)
    
    input_values = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, wave in enumerate(waveforms):
        length = wave.shape[0]
        input_values[i, :length] = wave
        attention_mask[i, :length] = 1
    
    region_labels = torch.tensor([b["region_id"] for b in batch], dtype=torch.long)
    gender_labels = torch.tensor([b["gender_id"] for b in batch], dtype=torch.long)
    coords = torch.tensor([b["coords"] for b in batch], dtype=torch.float32)
    
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "region_labels": region_labels,
        "gender_labels": gender_labels,
        "coords": coords
    }
