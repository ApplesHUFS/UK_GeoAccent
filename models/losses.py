"""
losses.py
Multi-task loss with distance regularization
"""

import torch
import torch.nn as nn


class MultiTaskLossWithDistance(nn.Module):
    """
    Combined loss function:
    1. Cross-entropy for region classification
    2. Cross-entropy for gender classification (aux)
    3. Cosine distance loss: 예측된 지역 임베딩 <-> 실제 지역 임베딩
    """
    
    def __init__(
        self,
        region_weight=1.0,
        gender_weight=0.3,
        distance_weight=0.5
    ):
        super().__init__()
        
        self.region_weight = region_weight
        self.gender_weight = gender_weight
        self.distance_weight = distance_weight
        
        self.region_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        self.distance_criterion = nn.CosineEmbeddingLoss()
    
    def forward(self, outputs, region_labels, gender_labels):
        """
        Args:
            outputs: model forward의 출력 dict
            region_labels: 지역 레이블 - (B,)
            gender_labels: 성별 레이블 - (B,)
        
        Returns:
            total_loss, region_loss, gender_loss, distance_loss
        """
        
        # 1. Region classification loss
        region_loss = self.region_criterion(
            outputs['region_logits'],  # (B, num_regions)
            region_labels              # (B,)
        )
        
        # 2. Gender classification loss
        gender_loss = self.gender_criterion(
            outputs['gender_logits'],  # (B, num_genders)
            gender_labels              # (B,)
        )
        
        # 3. Cosine distance loss
        if outputs['geo_embedding'] is not None:
            predicted_geo = outputs['predicted_geo_embedding']  # (B, geo_dim)
            actual_geo = outputs['geo_embedding']               # (B, geo_dim)
            
            # target: +1 (similar)
            target = torch.ones(predicted_geo.size(0)).to(predicted_geo.device)  # (B,)
            distance_loss = self.distance_criterion(predicted_geo, actual_geo, target)
        else:
            distance_loss = torch.tensor(0.0).to(outputs['region_logits'].device)
        
        # Total loss
        total_loss = (
            self.region_weight * region_loss +
            self.gender_weight * gender_loss +
            self.distance_weight * distance_loss
        )
        
        return total_loss, region_loss, gender_loss, distance_loss