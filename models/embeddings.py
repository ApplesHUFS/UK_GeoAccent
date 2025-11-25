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
        
        # 입력 차원 확인 및 정규화
        if audio_features.dim() == 3:
            # (B, T, audio_dim) -> (B, audio_dim) - mean pooling
            audio_features = audio_features.mean(dim=1)
        
        if geo_embeddings.dim() == 3:
            # (B, T, geo_dim) -> (B, geo_dim) - mean pooling
            geo_embeddings = geo_embeddings.mean(dim=1)
        
        # 이제 audio_features: (B, audio_dim), geo_embeddings: (B, geo_dim)
        
        query = self.query(audio_features)  # (B, fusion_dim)
        key = self.key(geo_embeddings)      # (B, fusion_dim)
        value = self.value(geo_embeddings)  # (B, audio_dim)
        
        # Scaled dot-product attention (단순화된 버전)
        # query: (B, fusion_dim), key: (B, fusion_dim)
        # attention_scores: (B,)
        attention_scores = torch.sum(query * key, dim=1, keepdim=True) / self.scale  # (B, 1)
        
        attention_weights = torch.sigmoid(attention_scores)  # (B, 1)
        
        # Attention 적용
        geo_context = attention_weights * value  # (B, 1) * (B, audio_dim) -> (B, audio_dim)
        
        # Concat
        fused = torch.cat([audio_features, geo_context], dim=1)  # (B, audio_dim*2)
        fused_features = self.output(fused)  # (B, audio_dim*2) -> (B, audio_dim)
        
        return fused_features, attention_weights

class AttentionFusion(nn.Module):
    """
    Wav2Vec2 Feature와 GeoEmbedding 정보를 Cross-Attention으로 융합
    """
    
    def __init__(self, audio_dim=1024, geo_dim=256, fusion_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.geo_dim = geo_dim
        
        # Projection layers
        self.audio_proj = nn.Linear(audio_dim, audio_dim)
        self.geo_proj = nn.Linear(geo_dim, audio_dim)
        
        # Cross-attention: audio queries geo
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output = nn.Sequential(
            nn.Linear(audio_dim * 2, audio_dim),
            nn.LayerNorm(audio_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio_features, geo_embeddings):
        """
        Args:
            audio_features: (B, audio_dim) or (B, T, audio_dim)
            geo_embeddings: (B, geo_dim)
        
        Returns:
            fused_features: (B, audio_dim)
            attention_weights: (B, 1, 1)
        """
        
        # 차원 정규화
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)  # (B, 1, audio_dim)
        
        if geo_embeddings.dim() == 2:
            geo_embeddings = geo_embeddings.unsqueeze(1)  # (B, 1, geo_dim)
        
        # Projection
        audio_proj = self.audio_proj(audio_features)  # (B, T, audio_dim)
        geo_proj = self.geo_proj(geo_embeddings)      # (B, 1, audio_dim)
        
        # Cross-attention: audio attends to geo
        attended, attention_weights = self.cross_attention(
            query=audio_proj,      # (B, T, audio_dim)
            key=geo_proj,          # (B, 1, audio_dim)
            value=geo_proj         # (B, 1, audio_dim)
        )
        
        # Pooling
        audio_pooled = audio_proj.mean(dim=1)  # (B, audio_dim)
        attended_pooled = attended.mean(dim=1)  # (B, audio_dim)
        
        # Fusion
        fused = torch.cat([audio_pooled, attended_pooled], dim=1)  # (B, audio_dim*2)
        fused_features = self.output(fused)  # (B, audio_dim)
        
        # Attention weights 평균
        attention_weights = attention_weights.mean(dim=1, keepdim=True)  # (B, 1, 1)
        
        return fused_features, attention_weights