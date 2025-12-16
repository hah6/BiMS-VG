# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules for Deformable DETR.
包含实现特征提取网络的组件，支持多尺度特征提取和位置编码。
"""

from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    固定参数的批量归一化层（BatchNorm2d）。冻结批统计量和仿射参数，保证推理一致性。

    修改自 torchvision.misc.ops，在计算倒数平方根(rqsrt)前添加 epsilon 避免除零错误。
    支持 ResNet 系列模型。
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        # 注册不可训练的缓冲区（非参数）
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 移除训练时追踪的批次数量统计量
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # 重整张量形状以便广播计算（效率优化）
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps

        # 冻结BN计算：scale = weight * 1/sqrt(var + eps)
        scale = w * (rv + eps).rsqrt()
        # 冻结BN计算：bias = bias - mean * scale
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """基础主干网络封装，支持中间层特征返回和梯度控制"""

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # 冻结不需要训练层的参数
        for name, parameter in backbone.named_parameters():
            # 默认只训练layer2及以上（可配置），冻结浅层特征
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # 配置要返回的特征层
        if return_interm_layers:
            # 返回layer2/3/4（用于多尺度特征）
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]  # 各层对应的下采样率
            self.num_channels = [512, 1024, 2048]  # 各层的输出通道数
        else:
            # 仅返回最后一层layer4
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        # 创建中间层特征提取器
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        """前向传播，处理图像和掩码"""
        # 提取特征图（忽略掩码）
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        # 为每个特征层生成对应的掩码
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将原始掩码下采样到当前特征图大小
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)  # 封装特征图和掩码
        return out


class Backbone(BackboneBase):
    """ResNet主干网络封装，支持空洞卷积(dilation)"""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 使用冻结BN层
        norm_layer = FrozenBatchNorm2d
        # 加载预训练ResNet模型
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],  # 最后阶段是否使用空洞卷积
            pretrained=is_main_process(),  # 仅在主进程下载预训练权重
            norm_layer=norm_layer)
        # 仅支持50层以上ResNet（通道数硬编码）
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        # 如果使用空洞卷积，修正最后一层的步幅
        if dilation:
            self.strides[-1] = self.strides[-1] // 2  # 32->16（因为空洞卷积保持分辨率）


class Joiner(nn.Sequential):
    """组合主干网络和位置编码器的序列模块"""

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        # 继承backbone的多尺度信息
        self.strides = backbone.strides  # 各特征层下采样率
        self.num_channels = backbone.num_channels  # 各特征层通道数

    def forward(self, tensor_list: NestedTensor):
        """前向传播：提取特征并计算位置编码"""
        # 通过主干网络提取多尺度特征
        xs = self[0](tensor_list)  # self[0]是backbone
        out: List[NestedTensor] = []
        pos = []

        # 按层名排序特征图（确保顺序一致）
        for name, x in sorted(xs.items()):
            out.append(x)

        # 为每个尺度的特征图生成位置编码
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))  # self[1]是position_embedding

        # 返回特征图列表 + 位置编码列表
        return out, pos
"""out = {
    "0": NestedTensor([B, 256, H1, W1], mask1),  # layer1
    "1": NestedTensor([B, 512, H2, W2], mask2),  # layer2
}pos=[
    [B, 256, H1, W1],  # layer1 位置编码
    [B, 512, H2, W2],  # layer2 位置编码
    ...
]

"""

def build_backbone(args):
    """构建主干网络和位置编码的组合模型"""
    # 创建位置编码器
    position_embedding = build_position_encoding(args)
    # 是否训练主干网络
    train_backbone = args.lr_visu_cnn > 0
    # 是否需要中间层特征（多尺度特征）
    return_interm_layers = (args.num_feature_levels > 1)  # 特征层数>1表示多尺度

    # 创建ResNet主干（支持dilation）
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 组合主干网络和位置编码器
    model = Joiner(backbone, position_embedding)
    return model