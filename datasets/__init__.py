from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import GroundingDataset

def make_transforms(args,image_set,is_onestage=False):
    imsize = args.imsize  # 640
    if is_onestage:
        normalize = Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            #Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])
        return normalize

    if image_set == "train":
        scales=[]
        """设置目标尺寸列表 scales：aug_scale=True时，生成一组不同大小用于随机缩放（例如640, 608, 576等）。否则固定为单一大小。
        crop_prob表示随机裁剪的概率。"""
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize-32*i)
        else:
            scales=[imsize]
        if args.aug_crop:
            crop_prob=0.5
        else:
            crop_prob=0
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            #T.ColorJitter(0.4, 0.4, 0.4),  # 是不是不应该用，尤其对refcoco+这种含颜色的
            T.GaussianBlur(aug_blur=args.aug_blur),
            #T.RandomHorizontalFlip(),  # 有的文章说不应该RandomHorizontalFlip，会掉点
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])
    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])
    raise ValueError(f'unknown {image_set}')
def build_dataset(split, args):
    if args.model_type == "ResNet":
        return GroundingDataset(data_root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split=split,
                            transform=make_transforms(args, split),
                            max_query_len=args.max_query_len)


