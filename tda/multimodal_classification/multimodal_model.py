#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态分类模型：图像(CNN+SelfAttn) + 关键点序列(Transformer) + 交叉注意力融合
轻量化设计：使用 ResNet18 和 浅层 Transformer 防止过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import math

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class KeypointSequenceEncoder(nn.Module):
    """关键点序列编码器 (Transformer)"""
    
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度 (cx, cy, w, h, conf, class_id) = 6
            d_model: Transformer hidden dim (Lightweight: 64)
            nhead: 注意力头数 (Lightweight: 4)
            num_layers: Transformer Encoder层数 (Lightweight: 1)
        """
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=20)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_model) # Optional
        
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # [B, S, D]
        x = self.transformer_encoder(x)
        return x


class ImageEncoderWithSelfOutput(nn.Module):
    """图像编码器 + Self Attention"""
    def __init__(self, backbone_name='resnet18', pretrained=True, d_model=64):
        super().__init__()
        
        # 使用 ResNet18 作为轻量级backbone
        # features_only=True 返回特征金字塔，我们取最后一层
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(4,) # 只取最后一层 (stride 32)
        )
        
        # 获取特征维度 (ResNet18 layer4 channel=512)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            feat = features[0] # [B, C, H, W]
            self.in_channels = feat.shape[1]
            self.spatial_size = feat.shape[2] * feat.shape[3] # 7x7=49
            
        print(f"Backbone: {backbone_name}, Channels: {self.in_channels}, Spatial: {self.spatial_size}")
        
        # 投影到 unified dimension d_model
        self.proj = nn.Conv2d(self.in_channels, d_model, kernel_size=1)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.spatial_size + 100)
        
        # 视觉自注意力 (1 layer Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True
        )
        self.visual_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, x):
        # 1. CNN Feature Extract
        # timm features_only returns a list
        feats = self.backbone(x)[0] # [B, 512, 7, 7]
        
        # 2. Projection
        feats = self.proj(feats) # [B, 64, 7, 7]
        
        # 3. Flatten for Transformer [B, S, D]
        # [B, 64, 49] -> [B, 49, 64]
        feats = feats.flatten(2).transpose(1, 2)
        
        # 4. Add Positional Encoding
        feats = self.pos_encoder(feats.transpose(0, 1)).transpose(0, 1) # [B, S, D]
        
        # 5. Self Attention
        feats = self.visual_transformer(feats) # [B, 49, 64]
        
        return feats


class CrossAttention(nn.Module):
    """交叉注意力模块"""
    
    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        """
        query: [B, Lq, D]
        key_value: [B, Lk, D]
        """
        # MultiheadAttention
        output, _ = self.mha(query, key_value, key_value)
        return self.norm(query + self.dropout(output))


class MultiModalTransformerClassifier(nn.Module):
    """
    双分支Transformer模型
    """
    def __init__(
        self,
        num_classes=19,
        image_backbone='resnet18',
        pretrained=True,
        d_model=64,        # 融合维度 (Lightweight)
        fusion_dim=128
    ):
        super().__init__()
        
        # 1. 图像分支
        self.image_encoder = ImageEncoderWithSelfOutput(
            backbone_name=image_backbone,
            pretrained=pretrained,
            d_model=d_model
        )
        
        # 2. 关键点分支
        self.keypoint_encoder = KeypointSequenceEncoder(
            input_dim=6, # [x, y, w, h, conf, cls]
            d_model=d_model,
            nhead=4,
            num_layers=1,
            dim_feedforward=128
        )
        
        # 3. 交叉融合 (使用 1 层 cross attn)
        # KP 查询 Image
        self.cross_kp2img = CrossAttention(dim=d_model, num_heads=4)
        # Image 查询 KP
        self.cross_img2kp = CrossAttention(dim=d_model, num_heads=4)
        
        # 4. Explicit Stat Features Projection
        self.stat_input_dim = 8
        self.stat_proj_dim = 32
        self.stat_proj = nn.Linear(self.stat_input_dim, self.stat_proj_dim)
        
        # 5. 融合聚合
        # Input: (Transformer_KP_Pool + Transformer_Image_Pool) + Stat_Proj
        fusion_input_dim = (d_model * 2) + self.stat_proj_dim
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 6. 分类头
        self.feat_dim = fusion_dim
        
    def forward(self, images, seq_features, stat_features=None):
        """
        Args:
            images: [B, 3, H, W]
            seq_features: [B, S, 6]
            stat_features: [B, 8] (Optional)
        """
        # 1. Encode
        img_feats = self.image_encoder(images)       # [B, 49, D]
        kp_feats = self.keypoint_encoder(seq_features) # [B, 10, D]
        
        # 2. Cross Attention Fusion
        kp_fused = self.cross_kp2img(query=kp_feats, key_value=img_feats) # [B, 10, D]
        img_fused = self.cross_img2kp(query=img_feats, key_value=kp_feats) # [B, 49, D]
        
        # 3. Pooling
        kp_pool = kp_fused.mean(dim=1)  # [B, D]
        img_pool = img_fused.mean(dim=1) # [B, D]
        
        to_concat = [kp_pool, img_pool]
        
        # 4. Explicit Features Fusion
        if stat_features is not None:
            stat_emb = self.stat_proj(stat_features) # [B, 32]
            to_concat.append(stat_emb)
        else:
            # Fallback if not provided (should be provided in updated training loop)
            dummy_stat = torch.zeros(kp_pool.size(0), self.stat_proj_dim, device=kp_pool.device)
            to_concat.append(dummy_stat)

        # 5. Concat & Project
        combined = torch.cat(to_concat, dim=1) # [B, 2D + 32]
        feature = self.fusion_fc(combined) # [B, fusion_dim]
        
        return feature


# 适配 BPaCo 接口的 Wrapper
class BpacoTransformerBackbone(nn.Module):
    def __init__(self, backbone='resnet18', proj_dim=128, pretrained=True):
        super().__init__()
        # 内部包含了 Image 和 Keypoint Encoder
        self.core_model = MultiModalTransformerClassifier(
            image_backbone=backbone,
            pretrained=pretrained,
            d_model=64,       # 控制参数量
            fusion_dim=256    # 最终特征维度 (before projection)
        )
        
        self.feat_dim = 256
        
        # BPaCo Projection Head
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim),
        )

    def forward(self, x, seq_features, stat_features=None):
        # 提取融合特征
        f = self.core_model(x, seq_features, stat_features) # [B, 256]
        
        # 投影
        z = self.proj(f)
        z = F.normalize(z, dim=1)
        
        return f, z


if __name__ == "__main__":
    # Test
    model = BpacoTransformerBackbone(backbone='resnet18')
    img = torch.randn(2, 3, 224, 224)
    seq = torch.randn(2, 10, 6)
    
    f, z = model(img, seq)
    print(f"Feature shape: {f.shape}") # [2, 256]
    print(f"Proj shape: {z.shape}")    # [2, 128]
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params / 1e6:.2f} M")
