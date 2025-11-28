"""
data/dataset.py
Custom PyTorch Dataset for JSON+WAV format
"""

import os
import json
import torch
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset

from preprocessing.preprocessing import AudioPreprocessor


class EnglishDialectsDataset(Dataset):
    """
    Loads dataset from JSON metadata and WAV files.
    Memory-efficient approach.
    """

    def __init__(self, split='train', use_augment=False,
                 data_dir="./data/english_dialects", audio_sample_rate=16000, random_seed=42):
        super().__init__()
        self.split = split
        self.use_augment = use_augment
        self.audio_sample_rate = audio_sample_rate
        self.data_dir = os.path.join(data_dir, split)

        # JSON 메타데이터 로드
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"✅ Loaded {split} split: {len(self.metadata)} samples")

        self.preprocessor = AudioPreprocessor(
            sample_rate=self.audio_sample_rate,
            use_augment=use_augment
        )

    def _load_audio(self, audio_path):
        """
        Load audio from WAV file.
        """
        full_path = os.path.join(self.data_dir, audio_path)

        # soundfile로 오디오 로드
        audio_data, sr = sf.read(full_path)

        # Resample if needed
        if sr != self.audio_sample_rate:
            audio_data = librosa.resample(
                y=np.array(audio_data, dtype=np.float32),
                orig_sr=sr,
                target_sr=self.audio_sample_rate
            )

        # Convert to tensor and normalize
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        waveform = self.preprocessor.normalize_audio(waveform)

        return waveform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 오디오 로드
        waveform = self._load_audio(item['audio_path'])

        # 메타데이터에서 라벨 가져오기
        region_id = item['region_id']
        gender_id = item['gender_id']
        coords = (item['normalized_lat'], item['normalized_lon'])
        config_name = item['config_name']

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
