# -*- coding: utf-8 -*-

"""
Generic Image Transform utillities.
"""
import torch
import cv2
import random, math
import numpy as np
from collections.abc import Iterable

import torch.nn.functional as F
from torch.autograd import Variable
"""
OpenCV / NumPy 图像	[H, W, C]
PyTorch Tensor 图像  [C, H, W]
"""

class ResizePad:
    """
  定义了一个图像预处理类 ResizePad，其功能是将图像等比例缩放后再**居中填充（pad）**到指定大小（目标高度 h，宽度 w）。
    保持比例缩放，不会拉伸图像；
    输出尺寸总是固定的 (self.h, self.w)；
    填充色为黑色（值为 0）；
    适配彩色图像和灰度图像。
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):#isinstance检查一个对象是否属于指定的类或类型，判断变量size是否是int类型或可迭代类型（如列表、元组等）。
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)#缩放比例
        resized_h = int(np.round(h * scale))#round函数返回浮点数x的四舍五入值，返回一个整数。
        resized_w = int(np.round(w * scale))
        #计算上下左右的填充边距
        pad_h = int(np.floor(self.h - resized_h) / 2)#floor函数向下取整
        pad_w = int(np.floor(self.w - resized_w) / 2)
        #用 OpenCV 缩放图像
        resized_img = cv2.resize(img, (resized_w, resized_h))

        # if img.ndim > 2:
        if img.ndim > 2:
            new_img = np.zeros((self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)#img.shape[-1]获取图像的通道数
        else:
            resized_img = np.expand_dims(resized_img, -1)#如果是灰度图（2D），先扩展维度；
            new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)
        new_img[pad_h: pad_h + resized_h,
                pad_w: pad_w + resized_w, ...] = resized_img#将缩放后的图像粘贴到黑色背景中央。
        return new_img
class ResizePad_tensor:
    """
    ResizePad 类：将输入图像（Tensor）按比例缩放到目标尺寸内，
    并在不足部分用0填充，使得输出图像大小固定为指定的 (h, w)。

    输入：
        img: Tensor，形状为 [C, H, W]
    输出：
        Tensor，形状为 [C, target_h, target_w]
    """

    def __init__(self, size):
        # size 可以是 int（高和宽相等）或者 Iterable（高，宽）
        if isinstance(size, int):
            self.h = size
            self.w = size
        elif isinstance(size, Iterable):
            self.h, self.w = size
        else:
            raise TypeError(f"size 参数必须是 int 或 Iterable，当前类型是 {type(size)}")

    def __call__(self, img):
        """
        执行 Resize 和 Padding 操作
        img: Tensor，[C, H, W]
        """
        if not torch.is_tensor(img):
            raise TypeError("输入 img 必须是 torch.Tensor 类型")

        c, h, w = img.shape
        # 计算缩放比例，保证缩放后图像不超过目标尺寸
        scale = min(self.h / h, self.w / w)

        resized_h = int(round(h * scale))
        resized_w = int(round(w * scale))

        # 使用双线性插值缩放图片到 resized_h x resized_w
        # 需要先给 img 添加 batch 维度 [1, C, H, W]
        img_resized = F.interpolate(img.unsqueeze(0), size=(resized_h, resized_w), mode='bilinear', align_corners=False)
        img_resized = img_resized.squeeze(0)  # 去掉 batch 维度，变回 [C, resized_h, resized_w]

        # 创建一个全0的tensor，用来放填充后的图像，大小为目标尺寸
        output = torch.zeros((c, self.h, self.w), dtype=img.dtype, device=img.device)

        # 计算上下左右的padding（使图像居中）
        pad_h = (self.h - resized_h) // 2
        pad_w = (self.w - resized_w) // 2

        # 将缩放后的图像复制到output的中心区域
        output[:, pad_h:pad_h+resized_h, pad_w:pad_w+resized_w] = img_resized

        return output

class CropResize:
    """去除之前因 ResizePad 添加的 padding，并将图像恢复到目标尺寸。"""

    def __call__(self, img, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        im_h, im_w = img.data.shape[:2]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        # scale = torch.Tensor([[input_h / im_h, input_w / im_w]]).max()
        resized_h = int(np.round(im_h * scale))
        # resized_h = torch.round(im_h * scale)
        resized_w = int(np.round(im_w * scale))
        # resized_w = torch.round(im_w * scale)
        crop_h = int(np.floor(resized_h - input_h) / 2)
        # crop_h = torch.floor(resized_h - input_h) // 2
        crop_w = int(np.floor(resized_w - input_w) / 2)
        # crop_w = torch.floor(resized_w - input_w) // 2
        # resized_img = cv2.resize(img, (resized_w, resized_h))
        resized_img = F.upsample(
            img.unsqueeze(0).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear')

        resized_img = resized_img.squeeze().unsqueeze(0)

        return resized_img[0, crop_h: crop_h + input_h,
                           crop_w: crop_w + input_w]


class CropResize_tensor:
    """
    Remove padding and resize image to its original size.

    Args:
        size: Tuple[int, int] -> target size (H, W)
    """
    def __init__(self, size):
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise TypeError(f"Size must be (height, width), got {size}")
        self.input_h, self.input_w = size

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): shape [C, H, W]
        Returns:
            torch.Tensor: shape [C, input_h, input_w]
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if img.dim() != 3:
            raise ValueError(f"Expected [C, H, W] tensor, got {img.shape}")

        _, im_h, im_w = img.shape

        # 放大比例（与 ResizePad 中使用 min 不同，这里使用 max 用于去除 padding）
        scale = max(self.input_h / im_h, self.input_w / im_w)

        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))

        # 双线性插值缩放图像
        img_resized = F.interpolate(img.unsqueeze(0), size=(resized_h, resized_w),
                                    mode='bilinear', align_corners=False).squeeze(0)

        crop_top = (resized_h - self.input_h) // 2
        crop_left = (resized_w - self.input_w) // 2

        cropped_img = img_resized[:, crop_top: crop_top + self.input_h,
                                       crop_left: crop_left + self.input_w]
        return cropped_img

class ResizeImage:
    """将图像最长边缩放到指定 size，保持比例"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            Variable(img).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear').squeeze().data
        return out


class ResizeAnnotation:
    """将输入的“标注图”按比例缩放，使其最长边缩放到指定大小 size，保持比例不变。"""
    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            Variable(img).unsqueeze(0).unsqueeze(0),
            size=(resized_h, resized_w),
            mode='bilinear').squeeze().data
        return out


class ToNumpy:
    """Transform an torch.*Tensor to an numpy ndarray."""

    def __call__(self, x):
        return x.numpy()

def letterbox(img, mask, height, color=(123.7, 116.3, 103.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        # print(top, bottom, left, right)
        # input()
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)  # padded square
        # print(mask)
    return img, mask, ratio, dw, dh

def random_affine(img, mask, targets, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5), all_bbox=None):
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    if mask is not None:
        maskw = cv2.warpPerspective(mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST,
                                  borderValue=1)  # BGR order borderValue
    else:
        maskw = None

    # Return warped points also
    if type(targets)==type([1]):
        targetlist=[]
        for bbox in targets:
            targetlist.append(wrap_points(bbox, M, height, a))
        return imw, maskw, targetlist, M
    elif all_bbox is not None:
        targets = wrap_points(targets, M, height, a)
        for ii in range(all_bbox.shape[0]):
            all_bbox[ii,:] = wrap_points(all_bbox[ii,:], M, height, a)
        return imw, maskw, targets, all_bbox, M
    elif targets is not None:   ## previous main
        targets = wrap_points(targets, M, height, a)
        return imw, maskw, targets, M
    else:
        return imw

def wrap_points(targets, M, height, a):
    # n = targets.shape[0]
    # points = targets[:, 1:5].copy()
    points = targets.copy()
    # area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
    area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, 1).T

    # reject warped points outside of image
    np.clip(xy, 0, height, out=xy)
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    ## print(targets, xy)
    ## [ 56  36 108 210] [[ 47.80464857  15.6096533  106.30993434 196.71267693]]
    # targets = targets[i]
    # targets[:, 1:5] = xy[i]
    targets = xy[0]
    return targets   