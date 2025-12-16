import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    """用于验证阶段评估预测框的精度。"""
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)## 限制在图像范围 [0, 1]
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)#将 IoU ≥ 0.5 的预测视为命中（准确）

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    """用于测试阶段评估预测框的准确数量。"""
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num
