"""
models/losses.py
Multi-task loss with distance regularization
"""

import torch
import torch.nn as nn


class MultiTaskLossWithDistance(nn.Module):
    """
    Combined loss function:
    1. Cross-entropy for region classification
    2. Cross-entropy for gender classification (auxiliary)
    3. Cosine distance loss: predicted region embedding vs. true region embedding
    """
    
    def __init__(
        self,
        region_weight=1.0,
        gender_weight=0.1,
        distance_weight=0.05
    ):
        super().__init__()
        
        self.region_weight = region_weight
        self.gender_weight = gender_weight
        self.distance_weight = distance_weight
        
        self.region_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        self.distance_criterion = nn.CosineEmbeddingLoss()
    
    def forward(self, outputs, region_labels, gender_labels):
        # Sanity checks
        if outputs.get('region_logits') is None:
            raise ValueError('Missing region_logits in model outputs')
        if outputs.get('gender_logits') is None:
            raise ValueError('Missing gender_logits in model outputs')

        if outputs['region_logits'].dim() != 2:
            raise ValueError(f"region_logits must be 2D (B, C), got {outputs['region_logits'].shape}")
        if region_labels.dim() != 1:
            raise ValueError(f"region_labels must be 1D (B,), got {region_labels.shape}")
        if outputs['region_logits'].size(0) != region_labels.size(0):
            raise ValueError(f"Batch size mismatch between region_logits {outputs['region_logits'].size(0)} and region_labels {region_labels.size(0)}")
        
        # 1. Region classification loss
        region_loss = self.region_criterion(outputs['region_logits'], region_labels)
        
        # 2. Gender classification loss
        gender_loss = self.gender_criterion(outputs['gender_logits'], gender_labels)
        
        # 3. Cosine distance loss
        if outputs['true_geo_embedding'] is not None:
            predicted_geo = outputs['predicted_geo_embedding']
            actual_geo = outputs['true_geo_embedding']
            target = torch.ones(predicted_geo.size(0), device=predicted_geo.device)
            distance_loss = self.distance_criterion(predicted_geo, actual_geo, target)
        else:
            distance_loss = torch.tensor(0.0, device=outputs['region_logits'].device)
        
        # Total loss
        total_loss = (
            self.region_weight * region_loss +
            self.gender_weight * gender_loss +
            self.distance_weight * distance_loss
        )

        # Ensure tensor type and device
        if not torch.is_tensor(total_loss):
            total_loss = torch.tensor(float(total_loss), device=outputs['region_logits'].device)

        return total_loss, region_loss, gender_loss, distance_loss
