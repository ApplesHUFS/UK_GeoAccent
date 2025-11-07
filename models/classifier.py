"""
classifier.py
GeoAccentClassifier 메인 모델
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from embeddings import GeoEmbedding, AttentionFusion


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
            self._freeze_lower_layers(num_frozen_layers)
        
        # 2. Region Embedding
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
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # 4. Classification Heads
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),     # (B, hidden_dim) -> (B, 512)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),            # (B, 512) -> (B, 256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)     # (B, 256) -> (B, num_regions)
        )
        
        # Gender Classification
        self.gender_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),     # (B, hidden_dim) -> (B, 256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_genders)     # (B, 256) -> (B, num_genders)
        )
        
        self.region_to_embedding = nn.Linear(num_regions, geo_embedding_dim)  # (B, num_regions) -> (B, geo_dim)
    
    def _freeze_lower_layers(self, num_frozen_layers):
        """
        하위 N개 레이어 freeze (상위 레이어에 집중)
        """
        print(f"Freezing lower {num_frozen_layers} layers")
        
        for layer_idx in range(num_frozen_layers):
            for param in self.wav2vec2.encoder.layers[layer_idx].parameters():
                param.requires_grad = False
        
        print(f"Trainable layers: {24 - num_frozen_layers}/24")  # XLSR-53은 24 layers
    
    def forward(self, input_values, attention_mask=None, coordinates=None, gender=None):
        """
        Args:
            input_values: 오디오 파형 - (B, seq_length)
            attention_mask: padding mask - (B, seq_length)
            coordinates: Normalized (lat, lon) - (B, 2)
            gender: 성별 레이블 - (B,)
        
        Returns:
            dict: {
                'region_logits': (B, num_regions),
                'gender_logits': (B, num_genders),
                'pooled_audio': (B, hidden_dim),
                'geo_embedding': (B, geo_dim),
                'fused_features': (B, hidden_dim),
                'attention_weights': (B, 1),
                'predicted_geo_embedding': (B, geo_dim)
            }
        """
        
        # 1. Feature Extraction
        wav2vec_outputs = self.wav2vec2(
            input_values,  # (B, seq_length)
            attention_mask=attention_mask,  # (B, seq_length)
            output_hidden_states=False
        )
        
        hidden_states = wav2vec_outputs.last_hidden_state  # (B, T, hidden_dim)
        
        # Temporal mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()  # (B, T, hidden_dim)
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)  # (B, hidden_dim)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, hidden_dim)
            pooled_audio = sum_hidden / sum_mask  # (B, hidden_dim)
        else:
            pooled_audio = hidden_states.mean(dim=1)  # (B, hidden_dim)
        
        # 2. Region Embedding
        if coordinates is not None:
            geo_embedding = self.geo_embedding(coordinates)  # (B, 2) -> (B, geo_dim)
            
            # 3. Attention Fusion
            fused_features, attention_weights = self.attention_fusion(
                pooled_audio,  # (B, hidden_dim)
                geo_embedding  # (B, geo_dim)
            )  # -> (B, hidden_dim), (B, 1)
        else:
            geo_embedding = None
            fused_features = pooled_audio  # (B, hidden_dim)
            attention_weights = None
        
        # Dropout
        fused_features = self.dropout(fused_features)  # (B, hidden_dim)
        
        # 4. Classification
        region_logits = self.region_classifier(fused_features)  # (B, hidden_dim) -> (B, num_regions)
        gender_logits = self.gender_classifier(fused_features)  # (B, hidden_dim) -> (B, num_genders)
        
        # 5. Predicted Geographic Embedding (for distance loss)
        region_probs = torch.softmax(region_logits, dim=-1)  # (B, num_regions)
        predicted_geo_embedding = self.region_to_embedding(region_probs)  # (B, num_regions) -> (B, geo_dim)
        
        return {
            'region_logits': region_logits,        # (B, num_regions)
            'gender_logits': gender_logits,        # (B, num_genders)
            'pooled_audio': pooled_audio,          # (B, hidden_dim)
            'geo_embedding': geo_embedding,        # (B, geo_dim) or None
            'fused_features': fused_features,      # (B, hidden_dim)
            'attention_weights': attention_weights,  # (B, 1) or None
            'predicted_geo_embedding': predicted_geo_embedding  # (B, geo_dim)
        }
