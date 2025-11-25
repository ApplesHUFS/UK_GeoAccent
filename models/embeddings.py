"""
models/embeddings.py
Geographic coordinate and attention fusion modules
"""

import torch
import torch.nn as nn
import math


class GeoEmbedding(nn.Module):
    """
    Converts geographic coordinates (lat, lon) to embedding vectors
    """
    
    def __init__(self, embedding_dim=256, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, coordinates):
        """
        Args:
            coordinates: Normalized (lat, lon) - (B, 2)
        Returns:
            embeddings: (B, embedding_dim)
        """
        return self.mlp(coordinates)


class AttentionFusion(nn.Module):
    """
    Fuses Wav2Vec2 features with geographic embeddings via attention
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
        # Normalize dimensions
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        if geo_embeddings.dim() == 2:
            geo_embeddings = geo_embeddings.unsqueeze(1)
        
        # Projection
        audio_proj = self.audio_proj(audio_features)
        geo_proj = self.geo_proj(geo_embeddings)
        
        # Cross-attention: audio attends to geo
        attended, attention_weights = self.cross_attention(
            query=audio_proj,
            key=geo_proj,
            value=geo_proj
        )
        
        # Pooling
        audio_pooled = audio_proj.mean(dim=1)
        attended_pooled = attended.mean(dim=1)
        
        # Fusion
        fused = torch.cat([audio_pooled, attended_pooled], dim=1)
        fused_features = self.output(fused)
        
        # Average attention weights
        attention_weights = attention_weights.mean(dim=1, keepdim=True)
        
        return fused_features, attention_weights
