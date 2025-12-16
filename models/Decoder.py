import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableDecoderLayer(nn.Module):
    """
    多尺度动态解码器 - 精确适配输入形状
    输入形状：
      text_queries: (L, B, C)  文本查询序列
      text_pos: (L, B, C)       文本位置编码
      ref_point: (B, 2)         参考点坐标
      sampling_query: (B, C)     采样查询向量
      ms_feats: list of (B, C, H, W) 多尺度视觉特征
      ms_pos: list of (B, C, H, W)   多尺度位置编码
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 in_points,  # 每个尺度的采样点数
                 feature_map_sizes,  # 各尺度特征图尺寸 [(H1,W1), (H2,W2), ...]
                 dropout=0.1,
                 dim_feedforward=2048,
                 uniform_grid=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_points = in_points
        self.feature_map_sizes = feature_map_sizes
        self.num_scales = len(feature_map_sizes)
        self.uniform_grid = uniform_grid

        # ===== 动态采样模块 =====
        # 每尺度独立的偏移量生成器
        self.offset_generators = nn.ModuleList([
            # 关键修正：使用当前尺度的采样点数，而不是整个列表
            nn.Linear(embed_dim, in_points[i] * 2)
            for i in range(self.num_scales)
        ])

        # 采样查询更新模块
        self.update_sampling_query = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        if self.uniform_grid:
            self.initial_grids = nn.ParameterList()
            for i, (H, W) in enumerate(feature_map_sizes):
                # 获取当前尺度的采样点数（已知是平方数）
                P = in_points[i]
                grid_size = int(math.sqrt(P))  # 网格边长

                # 生成均匀网格坐标（归一化到[0.5/grid_size, 1-0.5/grid_size]）
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size),
                    torch.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size),
                    indexing='ij'
                )

                # 直接展平为 (P, 2) 的网格坐标
                grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
                self.initial_grids.append(nn.Parameter(grid.unsqueeze(0)))  # (1, P, 2)
        # ===== 新增：视觉自注意力模块 =====
        # 在采样点上先做 self-attention 来增强局部上下文
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout_self = nn.Dropout(dropout)
        self.norm_self = nn.LayerNorm(embed_dim)
        # ===== 跨模态Transformer =====
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)  # attention 后加 Dropout
        # 4. 标准化层与FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
    def dynamic_sampling(self, stage_index, sampling_query, ref_point, feature_map, pos_map):
        """
        形状感知的动态采样
        输入:
          stage_index: 当前解码阶段索引
          sampling_query: (B, C)
          ref_point: (B, 2)
          feature_map: (B, C, H, W)
          pos_map: (B, C, H, W)
        输出:
          features: (B, C, P)
          pos_emb: (B, C, P)
        """
        B = sampling_query.size(0)
        # 获取当前阶段的采样点数
        current_points = self.in_points[stage_index]  # 使用当前尺度的点数
        # 动态采样点生成
        if self.uniform_grid and stage_index == 0:
            # 使用初始网格 (B, P, 2)
            sampled_points = self.initial_grids[stage_index].clone().repeat(B, 1, 1)
        else:
            # 生成偏移量 (B, P*2) -> (B, P, 2)
            xy_offsets = self.offset_generators[stage_index](sampling_query)
            xy_offsets = xy_offsets.view(B, current_points, 2)

            # 应用偏移 (B, P, 2)
            sampled_points = ref_point.unsqueeze(1) + xy_offsets

        # 归一化到[-1,1]范围
        sampled_points_norm = sampled_points * 2 - 1  # (B, P, 2)

        # 网格采样要求 (B, H_out, W_out, 2)
        grid = sampled_points_norm.unsqueeze(1)  # (B, 1, P, 2)

        # 双线性采样特征
        sampled_features = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # (B, C, 1, P)

        # 双线性采样位置编码
        sampled_pos = F.grid_sample(
            pos_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # (B, C, 1, P)

        return sampled_features.squeeze(2), sampled_pos.squeeze(2)  # (B, C, P)

    def forward(self, text_queries, text_pos_embed, ref_point, sampling_query, ms_feats, ms_pos, stage_index):
        """
        输入形状:
          text_queries: (L, B, C)
          text_pos: (L, B, C)
          ref_point: (B, 2)
          sampling_query: (B, C)
          ms_feats: list of (B, C, H, W)
          ms_pos: list of (B, C, H, W)
          stage_index: 当前解码阶段索引
        """
        B = text_queries.size(1)
        sampled_feats = []  # 存储 (P, B, C) 特征
        sampled_pos = []  # 存储 (P, B, C) 位置编码

        # ===== 多尺度采样 =====
        for scale_idx in range(self.num_scales):
            # 执行动态采样
            features, pos_emb = self.dynamic_sampling(
                scale_idx,
                sampling_query,
                ref_point,
                ms_feats[scale_idx],
                ms_pos[scale_idx]
            )  # (B, C, P)

            # 调整维度: (P, B, C)
            sampled_feats.append(features.permute(2, 0, 1))
            sampled_pos.append(pos_emb.permute(2, 0, 1))

        # ===== 特征融合 =====
        S = torch.cat(sampled_feats, dim=0)  # (∑P, B, C)
        P = torch.cat(sampled_pos, dim=0)  # (∑P, B, C)

        # 在拼好后先做一次 self-attention
        sa_out, _ = self.self_attn(S + P, S + P, S)
        sa_out = self.dropout_self(sa_out)
        S = self.norm_self(S + sa_out)

        # ========== 阶段3: 文本引导跨模态注意力 ==========
        query = text_queries + text_pos_embed  # (N_l, B, C)
        key = S + P  # (∑P_l, B, C)
        value = S  # (∑P_l, B, C)
        attn_output, _ = self.cross_attn(query, key, value)
        attn_output = self.dropout_attn(attn_output)
        text_updated = self.norm1(text_queries + attn_output)
        # ========== 阶段4: 前馈网络增强 ==========
        ffn_out = self.ffn(text_updated)
        text_out = self.norm2(text_updated + ffn_out)

        # ===== 动态参数更新 =====
        # 文本特征池化 (B, C)
        pooled_text = text_updated.mean(dim=0)

        # 更新参考点 (B, 2)
        delta_ref = torch.sigmoid(pooled_text[:, :2]) - 0.5
        new_ref = torch.clamp(ref_point + delta_ref, 0, 1)

        # 更新采样查询 (B, C)
        new_sampling_q = self.update_sampling_query(
            torch.cat([pooled_text, sampling_query], dim=-1)
        )

        return text_out, new_ref, new_sampling_q
class MLP(nn.Module):
    """多层感知机，用于回归 head"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim]*(num_layers-1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                     for i in range(len(dims)-1)])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers)-1 else layer(x)
        return x