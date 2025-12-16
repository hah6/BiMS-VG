import torch
from torchvision.ops.boxes import box_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    计算两个边框集合之间的 IoU（交并比），
    输入：
    box1, box2: shape 均为 [N, 4]，代表 N 个边框
    x1y1x2y2: 控制是否输入格式是 [x1, y1, x2, y2]（左上-右下角坐标）或 [x_center, y_center, w, h]
    返回：
    每对 box 的 IoU，shape 为 [N]
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def xywh2xyxy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """将 box 从 [x_center, y_center, w, h] 转换为 [x1, y1, x2, y2] 格式"""
    x_c, y_c, w, h = x.unbind(-1)
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)

    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h

    # 避免混精度或数值误差导致的坐标反转
    x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
    y1, y2 = torch.min(y1, y2), torch.max(y1, y2)

    return torch.stack([x1, y1, x2, y2], dim=-1)
def xyxy2xywh(x):
    """将 [x1, y1, x2, y2] 转换为 [x_center, y_center, w, h]"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """功能：计算两组 box 之间的两两 IoU（即得到 N×M 的 IoU 矩阵）
        输入：
        boxes1: [N, 4]
        boxes2: [M, 4]
        返回：
        iou: [N, M] 的 pairwise IoU 矩阵
        union: [N, M] 的并集面积矩阵"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算 Generalized IoU，输入需为 [x1, y1, x2, y2] 格式的 box
    输出 [N, M] 的 pairwise GIoU 矩阵
    """
    boxes1 = boxes1.clone()
    boxes2 = boxes2.clone()

    # 安全检查：如果存在无效 box（右下小于左上），直接返回零矩阵
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all() or not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        N, M = boxes1.shape[0], boxes2.shape[0]
        return torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)

    # IoU 与 Union
    iou, union = box_iou(boxes1, boxes2)  # shape [N, M]

    # 包含两个 box 的最小框区域
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # left top [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # right bottom [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    return iou - (area - union) / area  # Generalized IoU