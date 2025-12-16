# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
#from models.clip import clip
from torch.cuda.amp import autocast  # 新增

import math
import sys
import torch
from torch.cuda.amp import autocast
from typing import Iterable
from pathlib import Path
from utils.misc import NestedTensor
from fvcore.nn import FlopCountAnalysis
def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, max_norm: float = 0, scaler=None):
    """
    训练一个 epoch，支持 AMP、梯度裁剪、多卡 loss 同步。
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    print(f"数据加载器长度：{len(data_loader)}")

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img_data, text_data, target = batch


        # 发送到 GPU
        img_data = img_data.to(device)
        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # 前向传播和 loss 计算放在 autocast 上下文中（启用 AMP）
        with autocast():
            output= model(img_data, text_data)
            loss_dict = loss_utils.trans_vg_loss(output, target)
            loss = sum(loss_dict.values())

        # 检查 loss 是否有效
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss = {loss_value}, 停止训练")
            print(loss_dict)
            sys.exit(1)

        # 反向传播（scaler 支持 AMP）
        scaler.scale(loss).backward()

        # 可选：梯度裁剪（防止梯度爆炸）
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # 更新模型参数
        scaler.step(optimizer)
        scaler.update()

        # 日志记录（需 reduce 所有进程）
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        loss_value_reduced = sum(loss_dict_reduced_unscaled.values()).item()

        metric_logger.update(loss=loss_value_reduced, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 多卡同步日志
    metric_logger.synchronize_between_processes()
    print("平均统计指标:", metric_logger)

    # 返回每个指标的 epoch 级别平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    """
    验证函数：评估模型在验证集上的 loss、mIoU、accu。
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target = batch

        # 获取 batch size（兼容不同输入结构）
        batch_size = img_data.tensors.size(0) if args.model_type == "ResNet" else img_data.size(0)

        # 发送数据到设备
        img_data = img_data.to(device)
        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        # 如果需要支持 CLIP，可取消注释：
        # else:
        #     text_data = clip.tokenize(text_data).to(device)
        target = target.to(device)

        # 模型推理
        outputs = model(img_data, text_data)

        # loss 计算
        loss_dict = loss_utils.trans_vg_loss(outputs, target)
        loss = sum(loss_dict.values())

        # 多卡间 loss 同步（平均）
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        loss_value = sum(loss_dict_reduced_unscaled.values()).item()

        # 指标评估（IoU / accuracy）
        miou, accu = eval_utils.trans_vg_eval_val(outputs, target)

        # 更新指标日志
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)  # 平均 IoU
        metric_logger.update_v2('accu', accu, batch_size)              # IoU > 0.5 的比例

    # 多 GPU 同步日志
    metric_logger.synchronize_between_processes()

    # 打印平均结果
    print("Averaged stats:", metric_logger)

    # 返回各项指标的平均值（字典形式）
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        #else:
            #text_data = clip.tokenize(text_data).to(device)
        target = target.to(device)
        output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    # dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    return accuracy
