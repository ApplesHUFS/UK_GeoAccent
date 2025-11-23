"""
models/classifier.py
GeoAccentClassifier 메인 모델
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from .embeddings import GeoEmbedding, AttentionFusion


class GeoAccentClassifier(nn.Module):
    """
    Wav2Vec2 XLSR-53 모델 기반 지역 억양 분류기
    
    아키텍처 구조:
    1. Feature Encoder: Wav2Vec2-XLSR-53
    2. Region Embedding: (lat, lon) -> dense vector
    3. Attention Fusion: Audio x Geography
    4. Classification Heads: Region / Gender
    5. Distance Loss: 예측 지역 임베딩 vs 실제 임베딩 거리
    """
    
    def __init__(
        self,
        model_name="facebook/wav2vec2-large-xlsr-53",
        num_regions=6,
        num_genders=2,
        hidden_dim=1024,
        geo_embedding_dim=256,
        fusion_dim=512,
        dropout=0.1,
        freeze_lower_layers=True,
        num_frozen_layers=16
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_genders = num_genders
        self.hidden_dim = hidden_dim
        
        # 1. Feature Encoder
        print(f'Loading pretrained model: {model_name}')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze_lower_layers:
            self._freeze_lower_encoder_layers(num_frozen_layers)
        
        # 2. Geographic Embedding
        self.geo_embedding = GeoEmbedding(
            embedding_dim=geo_embedding_dim,
            dropout=dropout
        )
        
        # 3. Attention Fusion
        self.attention_fusion = AttentionFusion(
            audio_dim=hidden_dim,
            geo_dim=geo_embedding_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # 4. Classification Heads
        self.region_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)
        )
        
        self.gender_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_genders)
        )
        
        # 5. Region Embedding Predictor (for distance loss)
        self.region_embedding_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, geo_embedding_dim)
        )
    
    def _freeze_lower_encoder_layers(self, num_layers):
        """Freeze lower transformer layers"""
        print(f'Freezing lower {num_layers} encoder layers')
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_values, attention_mask, coordinates):
        """
        Args:
            input_values: (B, T) 오디오 waveform
            attention_mask: (B, T) 실제 오디오 구간 마스크
            coordinates: (B, 2) 정규화된 (lat, lon)
        
        Returns:
            dict: {
                'region_logits': (B, num_regions),
                'gender_logits': (B, num_genders),
                'predicted_geo_embedding': (B, geo_embedding_dim),
                'true_geo_embedding': (B, geo_embedding_dim)
            }
        """
        # 1. Audio Feature Extraction
        wav2vec_out = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        audio_features = wav2vec_out.last_hidden_state  # (B, T, hidden_dim)
        
        # 2. Geographic Embedding
        geo_embedding = self.geo_embedding(coordinates)  # (B, geo_embedding_dim)
        
        # 3. Attention Fusion
        fused_features = self.attention_fusion(
            audio_features, 
            geo_embedding
        )  # (B, fusion_dim)
        
        # 4. Classification
        region_logits = self.region_classifier(fused_features)  # (B, num_regions)
        gender_logits = self.gender_classifier(fused_features)  # (B, num_genders)
        
        # 5. Predict Region Embedding (for distance loss)
        predicted_geo_emb = self.region_embedding_predictor(fused_features)
        
        return {
            'region_logits': region_logits,
            'gender_logits': gender_logits,
            'predicted_geo_embedding': predicted_geo_emb,
            'true_geo_embedding': geo_embedding
        }
