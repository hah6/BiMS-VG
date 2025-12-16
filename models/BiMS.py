from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .visual_model.detr import build_detr
from .language_model.bert import build_bert

#from .MultiScaleDeformableDecoder import MultiScaleDeformableDecoderLayer  # 新引入
from .Decoder import MultiScaleDeformableDecoderLayer
from .featurEnhancer import EfficientFeatureEnhancer,build_efficient_feature_enhancer
from .CCFF import AdaptiveFeatureFusion
class BiMS(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_dim = args.vl_hidden_dim#256
        divisor = 16 if args.dilation else 32
        # 单尺度视觉 token 数
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        # 文本 token 最大长度
        self.num_text_token = args.max_query_len

        # -------- Backbone & Encoder --------
        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        self.text_pos_embed = nn.Embedding(self.num_text_token, hidden_dim)


        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)


        # -------- 特征增强模块 --------
        self.feature_enhancer=build_efficient_feature_enhancer(hidden_dim, args)

        self.feature_map_sizes = [(80, 80), (60, 60), (20, 20)]
        # -------- 多尺度可变形 Decoder --------
        num_points = args.num_points_per_scale  # 例如: [64,32,16]
        num_scales = len(num_points)
        self.decoder_layers = nn.ModuleList([
            MultiScaleDeformableDecoderLayer(
                embed_dim=hidden_dim,
                num_heads=args.vl_nheads,
                in_points=num_points,  # 每个尺度的采样点数
                feature_map_sizes=self.feature_map_sizes,
                dropout=args.vl_dropout,
                dim_feedforward=args.vl_dim_feedforward,
                uniform_grid=args.uniform_grid
            ) for _ in range(args.stages)
        ])

        # 初始化参考点和采样查询，不在 uniform_grid 上初始化
        self.init_reference_point = nn.Embedding(1, 2)
        self.init_sampling_feature = nn.Embedding(1, hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.init_weights()

    def init_weights(self):
        # 初始化参考点为中心
        nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
        nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
        self.init_reference_point.weight.requires_grad = False

        # 初始化偏移量生成器
        for layer in self.decoder_layers:
            for generator in layer.offset_generators:
                nn.init.zeros_(generator.weight)
                nn.init.uniform_(generator.bias, -0.5, 0.5)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        # 1. Visual + Text 编码
        out, visu_pos ,ms_feats,ms_pos= self.visumodel(img_data)#ms_feats: list of (B, C, H_l, W_l)
        visu_mask, visu_src = out# (B, H*W), (H*W, B, channel)。

        visu_src = visu_src.permute(1,0,2)# (H*W, B, channel)-> (B, H*W, channel)
        B, C, H, W = visu_pos.shape
        spatial_size = H * W
        visu_pos = visu_pos.view(B, C, spatial_size)


        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()# (B, L, C), (B, L)。
        # 投影并调整形状到 (seq_len, batch, C)
        text_src = self.text_proj(text_src)# (B,L,C)

        for layer in self.feature_enhancer:
            visu_src,text_src=layer(visu_src,text_src,visu_mask,text_mask)
        l_pos=self.text_pos_embed.weight.unsqueeze(0).expand(text_src.size(0), -1, -1)#(batch, L, C)
        language_feat=text_src
        language_feat = language_feat.permute(1, 0, 2)  # (L, batch, C)
        l_pos = l_pos.permute(1, 0, 2)  # (L, batch, C)
        # 2. 初始化 Decoder 状态
        ref_point = self.init_reference_point.weight.repeat(bs, 1)  # (B, 2)
        sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)  # (B, C)
        # 替换C5
        ms_feats.pop()
        # 2. 将visu_src转换为匹配格式并追加（对应新特征输入）
        B, HW, C = visu_src.shape  # 假设visu_src形状为(B,H*W,256)
        H, W = int(HW ** 0.5), int(HW ** 0.5)  # 假设是方形特征图
        visu_reshaped = visu_src.permute(0, 2, 1).reshape(B, C, H, W)  # (B,256,H,W)

        # 3. 追加到多尺度特征列表（保持与图中蓝色"Image Features"相同结构）
        ms_feats.append(visu_reshaped)

        ms_feats,ms_pos=build_fpn2(ms_feats,ms_pos,256)
       # ms_feats, ms_pos = self.AFF(ms_feats, ms_pos, 256)
        # 3. 多尺度 Deformable Decoder
        for stage_idx, layer in enumerate(self.decoder_layers):
            language_feat, ref_point, sampling_query = layer(
                text_queries=language_feat,  # (L, B, C)
                text_pos_embed=l_pos,  # (L, B, C)
                ref_point=ref_point,  # (B, 2)
                sampling_query=sampling_query,  # (B, C)
                ms_feats=ms_feats,  # 多尺度特征列表
                ms_pos=ms_pos,  # 多尺度位置编码列表
                stage_index=stage_idx  # 当前阶段索引
            )

        text_mask = text_mask.bool()  # 确保是布尔类型
        text_valid = (~text_mask).float()  # (B, L), 1 for valid tokens

        # 转换为 (L, B, 1) 方便广播
        weights = text_valid.T.unsqueeze(-1)  # (L, B, 1)

        # 权重加权求和
        weighted_sum = (language_feat * weights).sum(dim=0)  # (B, C)
        weights_sum = weights.sum(dim=0) + 1e-6  # 防止除以 0
        text_out = weighted_sum / weights_sum  # (B, C)
        # 回归边框并归一化
        pred_box = self.bbox_embed(text_out).sigmoid()
        return pred_box

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
def build_fpn(ms_feats: List[torch.Tensor], out_channels: int = 256) -> List[torch.Tensor]:
    """
    函数式FPN实现
    输入:
        ms_feats: 多尺度特征列表 [c3, c4, c5] (低层到高层)
        out_channels: 输出通道数 (默认256)
    返回:
        List[Tensor]: 融合后的特征金字塔 [p3, p4, p5, p6]
    """
    assert len(ms_feats) == 3, "输入需要是[c3, c4, c5]三级特征"
    c3, c4, c5 = ms_feats

    # 1. 横向连接 (1x1卷积统一通道数)
    def lateral_conv(in_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_uniform_(conv.weight, a=1)
        nn.init.constant_(conv.bias, 0)
        return conv.to(c5.device)

    l_c5 = lateral_conv(c5.size(1))(c5)  # C5 -> P5
    l_c4 = lateral_conv(c4.size(1))(c4)  # C4 -> P4
    l_c3 = lateral_conv(c3.size(1))(c3)  # C3 -> P3

    # 2. 自上而下路径
    p5 = l_c5
    p4 = l_c4 + F.interpolate(p5, size=l_c4.shape[-2:], mode='nearest')
    p3 = l_c3 + F.interpolate(p4, size=l_c3.shape[-2:], mode='nearest')

    # 3. 平滑处理 (3x3卷积)
    def smooth_conv():
        conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(conv.weight, a=1)
        nn.init.constant_(conv.bias, 0)
        return conv.to(c5.device)

    p4 = smooth_conv()(p4)
    p3 = smooth_conv()(p3)

    # 4. 生成P6 (对P5下采样)
    p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)

    return [p3, p4, p5]
def build_fpn2(
    ms_feats: List[torch.Tensor],
    vpos: List[torch.Tensor],
    out_channels: int = 256
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    函数式FPN实现（增强版：支持位置编码）

    输入:
        ms_feats: 多尺度特征列表 [c3, c4, c5] (低层到高层)
        vpos:     每层对应的位置编码列表 [vpos3, vpos4, vpos5]
        out_channels: 输出通道数 (默认256)

    返回:
        (List[Tensor], List[Tensor]):
            - FPN融合后的特征金字塔 [p3, p4, p5]
            - 与之匹配的融合位置编码 [vp3, vp4, vp5]
    """
    assert len(ms_feats) == 3 and len(vpos) == 3, "输入需要是[c3, c4, c5]和[vpos3, vpos4, vpos5]三级列表"
    c3, c4, c5 = ms_feats
    v3, v4, v5 = vpos

    # 横向连接 (1x1卷积统一通道数)
    def lateral_conv(in_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_uniform_(conv.weight, a=1)
        nn.init.constant_(conv.bias, 0)
        return conv.to(c5.device)

    l_c5 = lateral_conv(c5.size(1))(c5)  # C5 -> P5
    l_c4 = lateral_conv(c4.size(1))(c4)  # C4 -> P4
    l_c3 = lateral_conv(c3.size(1))(c3)  # C3 -> P3

    # 自上而下路径
    p5 = l_c5
    p4 = l_c4 + F.interpolate(p5, size=l_c4.shape[-2:], mode='nearest')
    p3 = l_c3 + F.interpolate(p4, size=l_c3.shape[-2:], mode='nearest')

    # 平滑处理 (3x3卷积)
    def smooth_conv():
        conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(conv.weight, a=1)
        nn.init.constant_(conv.bias, 0)
        return conv.to(c5.device)

    p4 = smooth_conv()(p4)
    p3 = smooth_conv()(p3)

    # 生成 P6（可选，暂不输出）
    # p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)

    # === 处理位置编码（vpos） ===
    # 只做上采样对齐
    vp5 = v5
    vp4 = F.interpolate(v4, size=vp5.shape[-2:], mode='bilinear', align_corners=False)
    vp4 = vp4 + vp5  # 对齐再融合（可选）

    vp3 = F.interpolate(v3, size=vp4.shape[-2:], mode='bilinear', align_corners=False)
    vp3 = vp3 + vp4  # 再次融合（可选）


    vp3 = F.interpolate(v3, size=p3.shape[-2:], mode='bilinear', align_corners=False)
    vp4 = F.interpolate(v4, size=p4.shape[-2:], mode='bilinear', align_corners=False)
    vp5 = v5

    return [p3, p4, p5], [vp3, vp4, vp5]