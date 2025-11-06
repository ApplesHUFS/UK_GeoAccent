import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoProcessor
import math
from data_config import REGION_COORDS #근데 이 data config는 data 폴더 내에 있어. 이 코드는 models 폴더 내에 있고

class GeoEmbedding(nn.Module):
    """
    지리 좌표 (위도, 경도)를 임베딩 벡터로 변환
    """

    def __init__(self,embedding_dim=256, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(2,128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, coordinates):
        """
        Args:
            coordinates: Normalized (lat,lon) (batch_size, 2)  
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.mlp(coordinates)
    
class AttentionFusion(nn.Module):
    '''
    Wav2Vec2 Feature와 GeoEmbedding 정보를 Attention으로 융합
    '''
    def __init__(self, audio_dim=1024, geo_dim=256, fusion_dim=512, dropout=0.1):
        super().__init__()

        self.audio_dim=audio_dim
        self.geo_dim=geo_dim
        self.fusion_dim=fusion_dim

        #Attention
        self.query=nn.Linear(audio_dim, fusion_dim)
        self.key=nn.Linear(geo_dim, fusion_dim)
        self.value=nn.Linear(geo_dim, audio_dim)

        self.scale=math.sqrt(fusion_dim)

        self.output=nn.Sequential(
            nn.Linear(audio_dim*2, audio_dim), #concat
            nn.LayerNorm(audio_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio_features, geo_embeddings):
        """
        Args:
            audio_features: Pooled Wav2Vec2 Features - (batch_size, audio_dim)
            geo_embeddings: Geographic Embeddings - (batch_size, geo_dim) 

        Returns:
            fused_features: Attention-fused Features - (batch_size, audio_dim)
            attention_weights: Attention scores - (batch_size,1)
        """
        batch_size= audio_features.size(0)

        query=self.query(audio_features) # (B, fusion_dim)
        key=self.key(geo_embeddings)     # (B, fusion_dim)
        value=self.value(geo_embeddings) # (B, audio_dim)


        # Scaled dot-product attention
        attention_scores=torch.bmm(
            query.unsqueeze(1),  # (B,1,fusion_dim)
            key.unsqueeze(2)     # (B, fusion_dim, 1)
        ).squeeze(-1) / self.scale # (B, 1)

        attention_weights=torch.softmax(attention_scores, dim=1) # (B, 1)

        geo_context = attention_weights * value # (B, audio_dim)

        #concat
        fused=torch.cat([audio_features, geo_context], dim=1) # (B, audio_dim*2)
        fused_features = self.output(fused) # (B, audio_dim)

        return fused_features, attention_weights

class GeoAccentClassifier(nn.Module):
    """
    Wav2Vec2 XLSR-53 모델 기반 지역 억양 분류기

    아키텍처 구조:
        1. Feature Encoder: Wav2Vec2-XLSR-53
        2. Region Embedding: (lat, lon) -> dense vector
        3. Attention Fusion:: Audio x Geography
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
            dropout = 0.1,
            freeze_lower_layers=True,
            num_frozen_layers=8 #하위 8개 레이어 얼리기
            ):
        

        super().__init__()

        self.num_regions=num_regions
        self.num_genders=num_genders
        self.hidden_dim=hidden_dim

        # 1. Feature Encoder
        print(f'Loading pretrained model: {model_name}')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_lower_layers: #하위 8개 레이어 얼리기
            self._freeze_lower_layers(num_frozen_layers)
        
        # 2. Region Embedding
        self.geo_embedding=GeoEmbedding(
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

        # 4. Classfication Heads
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)
        )

        # Gender Classification
        self.gender_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(256, num_genders)
        )

        self.region_to_embedding = nn.Linear(num_regions, geo_embedding_dim)
    
    def _freeze_lower_layers(self, num_frozen_layers):
        """
        하위 N개 레이어 freeze (상위 레이어에 집중)
        """
        
        print(f"Freezing lower {num_frozen_layers} layers")

        # 이 부분에서 CNN 프리즈도 필요한가?

        for layer_idx in range(num_frozen_layers):
            for param in self.wav2vec2.encoder.layers[layer_idx].parameters():
                param.requires_grad = False

        print(f"Trainable layers: {12-num_frozen_layers}/12") #응? 이 부분 식이 왜 이러지
    
    def forward(self, input_values, attention_mask=None, coordinates=None, gender=None):
        """
        Args:
            input_values: 오디오 파형 - (batch_size, seq_length)
            attention_mask: padding mask - (batch_size, seq_length)
            coordinates: Normalized (lat, lon) - (batch_size, 2)
            gender: 성별 레이블 - (batch_size, )
        
        Returns:
            dict: {
                'region_logits': (batch_size, num_regions),
                'gender_logits': (batch_size, num_genders),
                'pooled_audio' : (batch_size, hidden_dim),
                'geo_embedding': (batch_size, geo_dim),
                'fused_features': (batch_size, hidden_dim),
                'attention_weights': (batch_size, 1),
                'predicted_geo_embedding': (batch_size, geo_dim)
            }
        """

        # 1. Feature Extraction
        wav2vec_outputs=self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        hidden_states = wav2vec_outputs.last_hidden_state

        # Temporal mean pooling
        if attention_mask is not None:
            mask_expanded=attention_mask.unsqeeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_audio = sum_hidden / sum_mask # (batch_size, hidden_dim)
        else:
            pooled_audio = hidden_states.means(dim=1)
        
        # 2. Region Embedding
        if coordinates is not None:
            geo_embedding = self.geo_embedding(coordinates) # (batch_size, geo_dim)

            # 3. Attention Fusion
            fused_features, attention_weights = self.attention_fusion(
                pooled_audio, geo_embedding
            )
        else:
            geo_embedding = None
            fused_features = pooled_audio
            attention_weights = None
        
        # dropout
        fused_features = self.dropout(fused_features)

        # 4. Classficiation
        region_logits = self.region_classifier(fused_features) # (B, num_regions)
        gender_logits = self.gender_classifier(fused_features) # (B, num_genders)

        # 5. Predicted Geographic Embedding (for distance loss)
        region_probs = torch.softmax(region_logits, dim=-1) # (B, num_regions)
        predicted_geo_embedding = self.region_to_embedding(region_probs) # (B, geo_dim)

        return {
            'region_logits': region_logits,
            'gender_logits': gender_logits,
            'pooled_audio': pooled_audio,
            'geo_embedding': geo_embedding,
            'fused_features': fused_features,
            'attention_weights': attention_weights,
            'predicted_geo_embedding': predicted_geo_embedding
        }
    

class MultiTaskLossWithDistance(nn.Module):
    """
    Combined loss function:
    1. Cross-entropy for region classfication
    2. Cross-entropy for gender classfication (aux)
    3. Distance loss: 예측된 지역 임베딩 <-> 실제 지역 임베딩
    """

    def __init__(
        self,
        region_weight=1.0,
        gender_weight=0.3,
        distance_weight=0.5,
        distance_metric='cosine'
    ):
        super().__init__()

        self.region_weight = region_weight
        self.gender_weight = gender_weight
        self.distance_weight = distance_weight
        self.distance_metric = distance_metric
        
        self.region_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()

        # Cosine embedding loss
        if distance_metric == 'cosine':
            self.distance_criterion = nn.CosineEmbeddingLoss()
        else:
            self.distance_criterion = nn.MSELoss()
        
    def forward(self, outputs, region_labels, gender_labels):
        """
        Args:
            outputs: model forward의 출력 dict
            region_labels: 지역 레이블 - (batch_size, )
            gender_labels: 성별 레이블 - (batch_size , )

        Returns:
            total_loss, region_loss, gender_loss, distance_loss
        """

        # 1. Region classification loss
        region_loss = self.region_criterion(
            outputs['region_logits'],
            region_labels
        )

        # 2. Gender classfication loss
        gender_loss = self.gender_criterion(
            outputs['gender_logits'],
            gender_labels
        )

        # 3. Distance loss
        if outputs['geo_embedding'] is not None:
            predicted_geo = outputs['predicted_geo_embedding'] # (B, geo_dim)
            actual_geo = outputs['geo_embedding']              # (B, geo_dim)

            if self.distance_metric == 'cosine':
                # Cosine embedding loss
                # target: +1 (similar), -1 (dissimilar)
                # 같은 지역끼리는 가깝게
                target = torch.ones(predicted_geo.size(0)).to(predicted_geo.device)
                distance_loss = self.distance_criterion(predicted_geo, actual_geo, target)
            else:
                # MSE loss
                distance_loss = self.distance_criterion(predicted_geo, actual_geo)
        else:
            distance_loss = torch.tensor(0.0).to(outputs['region_logits'].device)
        
        # Total loss
        total_loss = (
            self.region_weight * region_loss +
            self.gender_weight * gender_loss +
            self.distance_weight * distance_loss
        )
        
        return total_loss, region_loss, gender_loss, distance_loss
                