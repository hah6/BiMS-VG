# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_queries, train_backbone, train_transformer, aux_loss=False):
        """ Initializes the model.
        backbone: 图像特征提取网络，比如 ResNet。
        transformer: Transformer 编-解码结构，用于空间建模和目标匹配。
        num_queries: 查询向量数量（目标候选槽位），控制最多检测多少个物体（如 100）。
        train_backbone: 是否训练 backbone。
        train_transformer: 是否训练 transformer。
        aux_loss: 是否使用辅助损失（来自每一层 decoder 输出，用于辅助训练）。
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.backbone = backbone
        backbone.num_channels=2048  # 确保 backbone 输出通道数为 2048（ResNet 的最后一层特征通道数）。
        #self.num_channels = backbone.num_channels
        if self.transformer is not None:
            hidden_dim = transformer.d_model
            # 因为 Transformer 需要的是 [sequence_length, batch_size, d_model] 的输入，不能直接吃 CNN 的 [B, C, H, W]，所以用 1x1 卷积把 CNN 输出通道映射到 transformer.d_model
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)#[C_in, C_out, kernel_size=1]
        else:
            hidden_dim = backbone.num_channels

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if self.transformer is not None and not train_transformer:
            for m in [self.transformer, self.input_proj]:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.num_channels = hidden_dim

    def forward(self, samples: NestedTensor):
        """samples: NestedTensor 是一个特殊封装，包含：
        samples.tensor: 图像张量 [B, 3, H, W]
        samples.mask: mask（1 表示 padding）[B, H, W]

        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)#mask:1是 padding，0是有效区域
        features, pos = self.backbone(samples)  # 用 CNN 提取特征，同时输出每层的 positional embeddings（位置编码）。
        src, mask = features[-1].decompose()  # 只用最后一层的特征，decompose() 拆成 (tensor, mask)。
        assert mask is not None

        if self.transformer is not None:
            out = self.transformer(self.input_proj(src), mask, pos[-1], query_embed=None)#输出一个元组(mask,memory)
        else:
            out = [mask.flatten(1), src.flatten(2).permute(2, 0, 1)]
        src_list = [feature.tensors for feature in features]  # 提取所有图像特征层的张量列表。
        return out, pos[-1], src_list,pos
        #return out, pos[-1]

    """当前返回的是：
    Transformer 输出（query 的输出，或 memory，取决于实现）。
    最后一层位置编码（可能用于后续的 box decoder 或 matcher）。"""


def build_detr(args):
    backbone = build_backbone(args)
    train_backbone = args.lr_visu_cnn > 0
    train_transformer = args.lr_visu_tra > 0
    if args.detr_enc_num > 0:
        transformer = build_transformer(args)
    else:
        transformer = None

    model = DETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        train_backbone=train_backbone,
        train_transformer=train_transformer
    )
    return model

