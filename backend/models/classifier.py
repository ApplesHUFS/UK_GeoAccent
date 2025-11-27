"""
models/classifier.py
GeoAccentClassifier model
"""
import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from .embeddings import GeoEmbedding, AttentionFusion


class GeoAccentClassifier(nn.Module):
    """
    Wav2Vec2 XLSR-53 based regional accent classifier.
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
        use_fusion=True
    ):
        super().__init__()
        
        # Save configuration
        self.model_name = model_name
        self.num_regions = num_regions
        self.num_genders = num_genders
        self.hidden_dim = hidden_dim
        self.geo_embedding_dim = geo_embedding_dim
        self.fusion_dim = fusion_dim
        self.dropout = dropout
        self.num_frozen_layers = num_frozen_layers
        self.use_fusion = use_fusion
        
        # Audio feature encoder
        print(f'Loading pretrained model: {model_name}')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze_lower_layers:
            self._freeze_lower_encoder_layers(num_frozen_layers)
        
        # Geographic embedding
        self.geo_embedding = GeoEmbedding(
            embedding_dim=geo_embedding_dim,
            dropout=dropout
        )
        
        # Attention-based fusion
        self.attention_fusion = AttentionFusion(
            audio_dim=hidden_dim,
            geo_dim=geo_embedding_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # Classification heads
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)
        )
        
        self.gender_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_genders)
        )
        
        # Region embedding predictor
        self.region_embedding_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
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
            input_values: (B, T) audio waveform
            attention_mask: (B, T) mask for valid audio frames
            coordinates: (B, 2) normalized (lat, lon)
        
        Returns:
            dict: {
                'region_logits': (B, num_regions),
                'gender_logits': (B, num_genders),
                'predicted_geo_embedding': (B, geo_embedding_dim),
                'true_geo_embedding': (B, geo_embedding_dim),
                'attention_weights': (B, 1)
            }
        """
        # Audio feature extraction
        wav2vec_out = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        audio_features = wav2vec_out.last_hidden_state
        
        # Mean pooling
        if attention_mask is not None:
            feature_attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
                audio_features.shape[1],
                attention_mask,
                add_adapter=False
            )
            mask_expanded = feature_attention_mask.unsqueeze(-1)
            sum_features = torch.sum(audio_features * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            audio_features_pooled = sum_features / sum_mask
        else:
            audio_features_pooled = audio_features.mean(dim=1)
        
        # Geographic embedding
        geo_embedding = self.geo_embedding(coordinates)
        
        # Attention fusion
        if self.use_fusion: 
            fused_features, attention_weights = self.attention_fusion(
                audio_features_pooled,
                geo_embedding
            )
        else:
            fused_features = audio_features_pooled 
            attention_weights = None 

        # Classification
        region_logits = self.region_classifier(fused_features)
        gender_logits = self.gender_classifier(fused_features)
        
        # Predict region embedding
        predicted_geo_emb = self.region_embedding_predictor(fused_features)
        
        return {
            'region_logits': region_logits,
            'gender_logits': gender_logits,
            'predicted_geo_embedding': predicted_geo_emb,
            'true_geo_embedding': geo_embedding,
            'attention_weights': attention_weights
        }
    
    def get_config(self):
        """Return model configuration for checkpointing"""
        return {
            'model_name': self.model_name,
            'num_regions': self.num_regions,
            'num_genders': self.num_genders,
            'hidden_dim': self.hidden_dim,
            'geo_embedding_dim': self.geo_embedding_dim,
            'fusion_dim': self.fusion_dim,
            'dropout': self.dropout,
            'num_frozen_layers': self.num_frozen_layers
        }
