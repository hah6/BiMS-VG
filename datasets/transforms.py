import torch
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from utils.box_utils import xyxy2xywh
from utils.misc import interpolate
def crop(image, box, region):
    """根据给定的裁剪区域 region，对图像和对应的目标框（box）进行同步裁剪，并返回裁剪后的图像和框。"""
    cropped_image = F.crop(image, *region)

    i, j, h, w = region

    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i])
    cropped_box = torch.min(cropped_box.reshape(2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)

    return cropped_image, cropped_box
def resize_according_to_long_side(img, box, size):
    """根据图像的长边调整图像和目标框的大小。"""
    h, w = img.height, img.width
    ratio = float(size / float(max(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio

    return img, box
def resize_according_to_short_side(img, box, size):
    """根据图像的短边调整图像和目标框的大小。"""
    h, w = img.height, img.width
    ratio = float(size / float(min(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio

    return img, box
class Compose(object):
    """将多个变换组合在一起，按顺序应用于输入数据。"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n" + "  " + repr(t)
        format_string += "\n)"
        return format_string


class RandomBrightness(object):
    """RandomBrightness 是一个图像增强操作，在一定范围内随机调整图像亮度，常用于数据增强（Data Augmentation）阶段，提升模型的泛化能力。"""
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)#创建一个亮度增强器。
        img = enhancer.enhance(brightness_factor)
        return img


class RandomContrast(object):
    """一个图像数据增强类 RandomContrast，其作用是随机调整图像的对比度"""
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    """一个图像增强类 RandomSaturation，用于随机调整图像的饱和度（saturation）"""
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img
class ColorJitter(object):
    """一个图像增强类 ColorJitter，用于随机改变图像的亮度、对比度和饱和度。它模拟 torchvision.transforms.ColorJitter 的功能，但提供了自定义实现。"""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast   = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, input_dict):
        if random.random() < 0.8:
            image = input_dict['img']
            func_inds = list(np.random.permutation(3))#生成 [0, 1, 2] 的一个随机排列，用于随机顺序执行三种扰动。
            for func_id in func_inds:
                if func_id == 0:
                    image = self.rand_brightness(image)
                elif func_id == 1:
                    image = self.rand_contrast(image)
                elif func_id == 2:
                    image = self.rand_saturation(image)
            input_dict['img'] = image

        return input_dict


class GaussianBlur(object):
    """一个图像增强类 GaussianBlur，用于给图像添加随机高斯模糊"""
    def __init__(self, sigma=[.1, 2.], aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.

    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict['img'] = img

        return input_dict
class RandomHorizontalFlip(object):
    """一个数据增强变换类 RandomHorizontalFlip，用于以 50% 的概率对输入图像及其对应的目标框（box）和文本描述进行水平翻转"""
    def __call__(self, input_dict):
        if random.random() < 0.5:
            img = input_dict['img']
            box = input_dict['box']
            text = input_dict['text']

            img = F.hflip(img)
            text = text.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            h, w = img.height, img.width
            box = box[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])

            input_dict['img'] = img
            input_dict['box'] = box
            input_dict['text'] = text

        return input_dict


class RandomResize(object):
    """一个用于随机调整图像大小的数据增强类 RandomResize，其核心功能是根据给定的一组尺寸，从中随机选择一个尺寸，并按照长边或短边来缩放图像及其对应的目标框。"""
    def __init__(self, sizes, with_long_side=True):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.with_long_side = with_long_side

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        size = random.choice(self.sizes)
        if self.with_long_side:
            resized_img, resized_box = resize_according_to_long_side(img, box, size)
        else:
            resized_img, resized_box = resize_according_to_short_side(img, box, size)

        input_dict['img'] = resized_img
        input_dict['box'] = resized_box
        return input_dict


class RandomSizeCrop(object):
    """一个基于**随机尺寸裁剪（RandomSizeCrop）**的数据增强类。核心功能是从给定的最小和最大尺寸范围内，尝试随机裁剪图像，同时保证裁剪区域包含目标框的位置。"""
    def __init__(self, min_size: int, max_size: int, max_try: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try


    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])  # [i, j, target_w, target_h]
            box_xywh = xyxy2xywh(box)
            box_x, box_y = box_xywh[0], box_xywh[1]
            if  region[1] <= box_x <= region[1] + w and region[0] <= box_y <= region[0] + h:
                img, box = crop(img, box, region)
                input_dict['img'] = img
                input_dict['box'] = box
                return input_dict

        return input_dict


class RandomSelect(object):
    """从两个变换序列（transforms1 和 transforms2）中选择一个应用于输入数据"""
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, input_dict):
        text = input_dict['text']

        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        # 如果文本中含有任何方向词，直接返回 transforms1 处理结果
        for wd in dir_words:
            if wd in text:
                return self.transforms1(input_dict)
        # 否则，以概率p执行 transforms2，(1-p)执行 transforms1
        if random.random() < self.p:
            return self.transforms2(input_dict)
        else:
            return self.transforms1(input_dict)


class ToTensor(object):
    def __call__(self, input_dict):
        """用于把输入字典中的图像数据从PIL Image或NumPy数组转换为PyTorch张量（Tensor）。
        这个转换通常用于数据增强流水线的最后步骤，保证图像数据能够输入到PyTorch模型。
        """
        img = input_dict['img']

        img = F.to_tensor(img)
        input_dict['img'] = img

        return input_dict
class NormalizeAndPad(object):
    """一个图像归一化和填充的transform，常用于图像预处理，尤其是当模型需要固定大小输入时。"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640, aug_translate=False):
        self.mean = mean              # 归一化的均值，常用ImageNet均值
        self.std = std                # 归一化的标准差，常用ImageNet标准差
        self.size = size              # 目标尺寸（正方形边长）
        self.aug_translate = aug_translate  # 是否在填充时做随机平移数据增强

    def __call__(self, input_dict):
        img = input_dict['img']
        # 归一化，标准化到均值为0，方差为1的分布
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]  # 通道高宽 C,H,W
        dw = self.size - w    # 宽度差距
        dh = self.size - h    # 高度差距

        # 是否做随机平移（填充时随机选填充的起始位置）
        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)  # 这里-0.1是微调，防止四舍五入误差
            left = round(dw / 2.0 - 0.1)

        # 创建目标大小的空白图像，初始填充为0
        out_img = torch.zeros((3, self.size, self.size), dtype=torch.float)
        # 创建掩码，1表示填充区域，0表示有效区域
        out_mask = torch.ones((self.size, self.size), dtype=torch.int)

        # 将归一化后的图像内容复制到目标图像中（指定位置）
        out_img[:, top:top+h, left:left+w] = img
        out_mask[top:top+h, left:left+w] = 0  # 有效区域标记为0

        input_dict['img'] = out_img
        input_dict['mask'] = out_mask

        # 如果有目标框，调整框的坐标以匹配填充后的图像
        if 'box' in input_dict.keys():
            box = input_dict['box']
            box[0], box[2] = box[0] + left, box[2] + left  # x坐标平移
            box[1], box[3] = box[1] + top, box[3] + top    # y坐标平移

            h, w = out_img.shape[-2:]  # 更新宽高
            box = xyxy2xywh(box)       # 将xyxy格式转换成xywh格式
            box = box / torch.tensor([w, h, w, h], dtype=torch.float32)  # 归一化到[0,1]
            input_dict['box'] = box

        return input_dict
class MaskRISTransform:
    """
    MaskRIS 风格的图像 & 文本遮挡增强，可在 Compose 里直接使用。

    输入 sample: (img, text_tokens)
    输出: (img_masked, text_tokens_masked)
    """
    def __init__(self,
                 mask_ratio_img=0.5,
                 mask_ratio_text=0.15,
                 tokenizer=None,
                 mask_token_id=None,
                 patch_size=16):
        self.mask_ratio_img = mask_ratio_img
        self.mask_ratio_text = mask_ratio_text
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.patch_size = patch_size

    def __call__(self, sample):
        # 解包
        img, text_tokens = sample
        # 确保 img 是 Tensor
        if not torch.is_tensor(img):
            img = ToTensor()(img)

        # 1) 图像遮挡
        C, H, W = img.shape
        ph, pw = self.patch_size, self.patch_size
        nh, nw = H // ph, W // pw
        total = nh * nw
        mask_cnt = int(total * self.mask_ratio_img)
        patches = [(i, j) for i in range(nh) for j in range(nw)]
        to_mask = random.sample(patches, mask_cnt)
        img_masked = img.clone()
        for i, j in to_mask:
            h0, w0 = i*ph, j*pw
            img_masked[:, h0:h0+ph, w0:w0+pw] = 0

        # 2) 文本遮挡
        text_masked = text_tokens.clone()
        for idx in range(text_masked.size(0)):
            if random.random() < self.mask_ratio_text:
                r = random.random()
                if r < 0.8:
                    text_masked[idx] = self.mask_token_id
                elif r < 0.9 and self.tokenizer is not None:
                    text_masked[idx] = random.choice(
                        list(self.tokenizer.get_vocab().values())
                    )

        return img_masked, text_masked
