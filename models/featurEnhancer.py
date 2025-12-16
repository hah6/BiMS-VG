from typing import Optional, Tuple

import torch
import torch.nn as nn
from .fuse_modules import BiAttentionBlock


import torch
import torch.nn as nn
from .fuse_modules import BiAttentionBlock


class EfficientFeatureEnhancer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, droppath=0.0, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(embed_dim)
        self.dropout_self = nn.Dropout(dropout)

        self.bi_attn = BiAttentionBlock(
            v_dim=embed_dim,
            l_dim=embed_dim,
            embed_dim=dim_feedforward // 2,
            num_heads=num_heads // 2,
            dropout=dropout,
            drop_path=droppath,
        )


        self.ffn_visu = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward  ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward  , embed_dim)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward , embed_dim)
        )
        self.norm_ffn_visu = nn.LayerNorm(embed_dim)
        self.norm_ffn_text = nn.LayerNorm(embed_dim)

    def forward(self, visu_feat, text_feat, visu_mask, text_mask):
        """
        - v_pos: optional 位置编码 (B, N, C)，仅用于 query/key
        """

        # === 1. BiAttention ===
        visu_feat, text_feat = self.bi_attn(
            visu_feat,  # 加位置信息
            text_feat,
            attention_mask_v=visu_mask,
            attention_mask_l=text_mask
        )

        # === 2. Self-Attention（视觉） ===
        attn_out, _ = self.self_attn(
            query=visu_feat,
            key=visu_feat,
            value=visu_feat,  # 保持纯净 value
            key_padding_mask=visu_mask
        )
        attn_out = self.dropout_self(attn_out)
        visu_feat = self.norm_self(visu_feat + attn_out)

        # === 3. FFN ===
        visu_feat = self.norm_ffn_visu(visu_feat + self.ffn_visu(visu_feat))
        text_feat = self.norm_ffn_text(text_feat + self.ffn_text(text_feat))

        return visu_feat, text_feat

def build_efficient_feature_enhancer(hidden_dim, args):
    """
    创建一个 EfficientFeatureEnhancer 的 ModuleList，并返回可控位置编码版本
    """
    layers = nn.ModuleList([
        EfficientFeatureEnhancer(
            embed_dim=hidden_dim,
            num_heads=args.nheads,
            dropout=args.dropout,
            droppath=0.1,
            dim_feedforward=2048
        )
        for _ in range(args.feature_enhancer_layers)
    ])
    return layers
