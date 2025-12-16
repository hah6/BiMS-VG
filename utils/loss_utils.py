import torch
import numpy as np
import torch.nn.functional as F

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size

def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            # gi = coord_list[scale_ii][ii,0].long()
            # gj = coord_list[scale_ii][ii,1].long()
            # tx = coord_list[scale_ii][ii,0] - gi.float()
            # ty = coord_list[scale_ii][ii,1] - gj.float()
            gw = coord_list[scale_ii][ii,2]
            gh = coord_list[scale_ii][ii,3]

            anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            ## Get shape of gt box
            # gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # import pdb
            # pdb.set_trace()

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/best_grid), \
            x[1] / (args.anchor_imsize/best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii,0].long()
        gj = coord_list[best_scale][ii,1].long()
        tx = coord_list[best_scale][ii,0] - gi.float()
        ty = coord_list[best_scale][ii,1] - gj.float()
        gw = coord_list[best_scale][ii,2]
        gh = coord_list[best_scale][ii,3]
        tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3,0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii]//3][ii, best_n_list[ii]%3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
        gt_conf_list.append(target[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf

def trans_vg_loss_with_budget(batch_pred, batch_target, budget_analysis=None,
                              budget_loss_weights=None):
    """
    融合预算损失的视觉定位损失函数

    输入:
      batch_pred: Tensor [B, 4] 预测框坐标 [cx, cy, w, h]
      batch_target: Tensor [B, 4] 真实框坐标 [cx, cy, w, h]
      budget_analysis: dict 预算分析信息（来自解码器）
      budget_loss_weights: dict 各预算损失项的权重

    输出:
      losses: dict 包含所有损失项
    """
    # 默认预算损失权重
    if budget_loss_weights is None:
        budget_loss_weights = {
            'loss_budget_constraint': 0.05,  # 预算约束权重
            'loss_budget_sparsity': 0.025,  # 稀疏性权重
            'loss_budget_entropy': 0.01  # 熵正则权重
        }

    batch_size = batch_pred.shape[0]
    num_boxes = batch_size

    # ===== 原有的边框回归损失 =====
    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')

    # 防止AMP精度问题，转为float32计算GIoU
    batch_pred_32 = batch_pred.float()
    batch_target_32 = batch_target.float()

    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred_32),
        xywh2xyxy(batch_target_32)
    ))

    # 基础损失
    losses = {
        'loss_bbox': loss_bbox.sum() / num_boxes,
        'loss_giou': loss_giou.sum() / num_boxes
    }

    # ===== 融合预算损失 =====
    if budget_analysis is not None:
        for loss_name, loss_value in budget_analysis.items():
            weight = budget_loss_weights.get(loss_name, 0.01)
            losses[loss_name] = weight * loss_value

    return losses
def trans_vg_loss(batch_pred, batch_target):
    """
    用于目标检测任务的边框回归损失函数，适用于像 DETR 或其他基于 Transformer 的检测器，
    它同时计算了两种损失：L1 和 GIoU。

    batch_pred: Tensor of shape [B, 4]，预测框坐标 [cx, cy, w, h]
    batch_target: Tensor of shape [B, 4]，真实框坐标 [cx, cy, w, h]
    """
    batch_size = batch_pred.shape[0]
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')  # L1 损失

    # === 新增: 为防止 AMP 下精度问题导致 GIoU 计算报错，手动转为 float32 ===
    batch_pred_32 = batch_pred.float()
    batch_target_32 = batch_target.float()

    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred_32),  # 转为 float32 后再转坐标格式
        xywh2xyxy(batch_target_32)
    ))

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses
