"""
embeddings.py
지리 좌표와 Attention Fusion 임베딩 관련 모듈
"""

import torch
import torch.nn as nn
import math


class GeoEmbedding(nn.Module):
    """
    지리 좌표 (위도, 경도)를 임베딩 벡터로 변환
    """
    
    def __init__(self, embedding_dim=256, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),  # (B, 2) -> (B, 128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),  # (B, 128) -> (B, embedding_dim)
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, coordinates):
        """
        Args:
            coordinates: Normalized (lat, lon) - (B, 2)
        Returns:
            embeddings: (B, embedding_dim)
        """
        return self.mlp(coordinates)  # (B, 2) -> (B, embedding_dim)


class AttentionFusion(nn.Module):
    """
    Wav2Vec2 Feature와 GeoEmbedding 정보를 Attention으로 융합
    """
    
    def __init__(self, audio_dim=1024, geo_dim=256, fusion_dim=512, dropout=0.1):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.geo_dim = geo_dim
        self.fusion_dim = fusion_dim
        
        # Attention
        self.query = nn.Linear(audio_dim, fusion_dim)  # (B, audio_dim) -> (B, fusion_dim)
        self.key = nn.Linear(geo_dim, fusion_dim)      # (B, geo_dim) -> (B, fusion_dim)
        self.value = nn.Linear(geo_dim, audio_dim)     # (B, geo_dim) -> (B, audio_dim)
        
        self.scale = math.sqrt(fusion_dim)
        
        self.output = nn.Sequential(
            nn.Linear(audio_dim * 2, audio_dim),  # (B, audio_dim*2) -> (B, audio_dim)
            nn.LayerNorm(audio_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio_features, geo_embeddings):
        """
        Args:
            audio_features: Pooled Wav2Vec2 Features - (B, audio_dim)
            geo_embeddings: Geographic Embeddings - (B, geo_dim)
        
        Returns:
            fused_features: Attention-fused Features - (B, audio_dim)
            attention_weights: Attention scores - (B, 1)
        """
        batch_size = audio_features.size(0)  # B
        
        query = self.query(audio_features)  # (B, audio_dim) -> (B, fusion_dim)
        key = self.key(geo_embeddings)      # (B, geo_dim) -> (B, fusion_dim)
        value = self.value(geo_embeddings)  # (B, geo_dim) -> (B, audio_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.bmm(
            query.unsqueeze(1),  # (B, 1, fusion_dim)
            key.unsqueeze(2)     # (B, fusion_dim, 1)
        ).squeeze(-1) / self.scale  # (B, 1, 1) -> (B, 1)
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, 1)
        
        geo_context = attention_weights * value  # (B, 1) * (B, audio_dim) -> (B, audio_dim)
        
        # Concat
        fused = torch.cat([audio_features, geo_context], dim=1)  # (B, audio_dim*2)
        fused_features = self.output(fused)  # (B, audio_dim*2) -> (B, audio_dim)
        
        return fused_features, attention_weights
