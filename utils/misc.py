# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

# fix bug
version = torchvision.__version__.split('.')[1]
if int(version) < 7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

# original
# if float(torchvision.__version__[:3]) < 0.7:
#     from torchvision.ops import _new_empty_tensor
#     from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """用于记录并平滑输出训练过程中的指标（如loss），支持滑动平均与全局平均
    """
    """ 滑动窗口中位数（median）
    滑动窗口平均值（avg）
    全局加权平均（global_avg）
    当前值（value）
    最大值（max）"""

    def __init__(self, window_size=1, fmt=None):
        if fmt is None:
            fmt = "median={median:.4f} global_avg=({global_avg:.4f})"#保留浮点数四位小数
        self.deque = deque(maxlen=window_size)#Python 的双端队列，用于高效记录最近的数值
        self.total = 0.0#累计总值（用于计算全局平均）
        self.count = 1e-12#累计样本数（用于计算全局平均）
        self.fmt = fmt#字符串格式控制

    def update(self, value, n=1):#把一个新值 value 添加到滑动窗口中（deque）
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在多 GPU（分布式）训练时，同步各个进程的 count 和 total
        注意：deque（滑动窗口）不参与同步！
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self): #计算滑动窗口的中位数
        # import pdb
        # pdb.set_trace()
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self): #计算滑动窗口的平均值
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):  #计算全局平均值
        return self.total / self.count

    @property
    def max(self):  #计算滑动窗口的最大值
        return max(self.deque)

    @property
    def value(self):  #获取滑动窗口的最后一个值
        return self.deque[-1]
    
    def get_global_avg(self):  #获取全局平均值
        return self.total / self.count

    def __str__(self): #使用使用 print() 打印对象时，会调用此方法
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    这段代码定义了一个用于 分布式训练中跨进程通信 的实用函数 all_gather(data)，它可以在 多个 GPU 或进程之间收集任意 Python 对象，而不仅仅是张量。
    PyTorch 的 dist.all_gather() 只能收集 形状一致的张量，而这个函数通过序列化（pickle.dumps）和 padding，实现了任意 Python 对象的跨进程收集。
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    #如果只运行在单进程/单GPU，则直接返回当前 data。
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 将数据序列化为张量
    buffer = pickle.dumps(data) # 任意对象序列化成 bytes
    storage = torch.ByteStorage.from_buffer(buffer)# 转为 ByteStorage
    tensor = torch.ByteTensor(storage).to("cuda") # 再转成 CUDA Tensor

    # 获取每个进程张量的大小
    local_size = torch.tensor([tensor.numel()], device="cuda") # 当前数据的长度
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)# 收集每个进程的数据长度
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)#因为张量长度不一定一样，需要收集长度列表 size_list，再找出最大值。


    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    #为所有进程准备统一大小的张量（padding）
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    #如果当前进程的张量长度小于最大长度，则需要填充（padding）
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)#收集所有进程的数据
    #从收集到的 byte tensor 中，提取真实长度的数据并反序列化回 Python 对象。
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list#data_list 是一个列表，包含所有进程传入的 data。


def reduce_dict(input_dict, average=True):
    """
    这个函数 reduce_dict 是在 分布式训练（Distributed Training）中 用来 对多个进程的字典中的值进行汇总（求和或平均） 的实用函数。
    它的目的是：让所有进程都拥有同样的、全局一致的统计值（例如 loss、metrics 等）。
    """
    """
    输入：
    input_dict：每个进程上的一个字典，里面的值是 Tensor（例如 {"loss": tensor(0.8), "acc": tensor(0.5)}）
    average：是否对所有进程的值进行平均（否则就求和）
    输出：返回一个 每个 key 都是全局汇总（求和或平均）后的字典
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    """这个 MetricLogger 类是 PyTorch 分布式训练或大型训练任务中常用的 训练指标记录与输出工具"""

    def __init__(self, delimiter="\t"):#功能：初始化一个日志记录器
        self.meters = defaultdict(SmoothedValue)#self.meters：用于记录多个指标（比如 loss、accuracy）。每个指标用 SmoothedValue 类型来表示，具有平滑、平均等统计功能。
        self.delimiter = delimiter#delimiter：控制打印日志时各项之间的分隔符，默认是制表符 \t。

    def update(self, **kwargs):
        # import pdb
        # pdb.set_trace()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_v2(self, key, value, num):
        self.meters[key].update(value, num)
        # for k, v in kwargs.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v.item()
        #     assert isinstance(v, (float, int))
        #     self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))#str(meter) 会调用 SmoothedValue.__str__() 方法，返回一个形如 "0.1234 (0.5678)" 的字符串（你在前面设置的格式 "{median:.4f} ({global_avg:.4f})"）。
            )
        return self.delimiter.join(loss_str)#这会把所有指标的字符串拼接成一个整体字符串，最终打印输出用

    def synchronize_between_processes(self):#如果使用多 GPU 分布式训练，需要让各 GPU 之间的指标同步。
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    #手动添加一个新的统计指标；
    def add_meter(self, name, meter):
        self.meters[name] = meter
    """在训练/推理等循环中定期打印日志信息（如迭代进度、时间、显存、损失等）
    iterable：你要遍历的数据（比如 DataLoader）
    print_freq：每隔几步打印一次日志
    header：日志前缀标题（可选）"""
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')#记录每一步迭代耗时
        data_time = SmoothedValue(fmt='{avg:.4f}')#记录每次加载数据的耗时（yield 之前
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'#根据数据集长度，决定打印对齐格式，比如 :4d
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)# 加载数据耗时
            yield obj  # 真正执行你的代码（外部训练逻辑）
            iter_time.update(time.time() - end)# 执行代码耗时
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))#估算 ETA（剩余时间）
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),## 会调用 MetricLogger.__str__()
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


# def collate_fn(raw_batch):
#     raw_batch = list(zip(*raw_batch))
#     img_data = nested_tensor_from_tensor_list(raw_batch[0])
#     word_id = torch.tensor(raw_batch[1])
#     word_mask = torch.tensor(raw_batch[2])
#     text_data = NestedTensor(word_id, word_mask)
#     bbox = torch.tensor(raw_batch[3])
#     batch = [img_data, text_data, bbox]
#     return tuple(batch)


def collate_fn(raw_batch):
    """用于将每个 batch 中的样本打包成 tensor 或 NestedTensor 结构，供模型使用。"""
    """raw_batch是一个 batch 的原始列表,(img1, img_mask1, word_ids1, word_mask1, bbox1),
    (img2, img_mask2, word_ids2, word_mask2, bbox2),"""
    for i, sample in enumerate(raw_batch):
        assert len(sample) == 5, f"样本 {i} 不包含 5 个元素，而是 {len(sample)}: {sample}"
    raw_batch = list(zip(*raw_batch))#这行的作用是将上面结构“转置”，变成 5 个列表
    """raw_batch = [
      [img1, img2, ...],        # 图像列表
      [img_mask1, img_mask2],   # 图像 mask 列表
      [word_ids1, word_ids2],   # 文本 token id 列表
      [word_mask1, word_mask2], # 文本 mask 列表
      [bbox1, bbox2]            # bbox 列表
    ]"""
    img = torch.stack(raw_batch[0])  # 将图像堆叠成 tensor [B, C, H, W]
    img_mask = torch.as_tensor(np.array(raw_batch[1]))  # 图像 mask
    img_data = NestedTensor(img, img_mask)  # 包成 NestedTensor
    word_id = torch.stack([torch.as_tensor(w, dtype=torch.long) for w in raw_batch[2]])
    word_mask = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in raw_batch[3]])

    text_data = NestedTensor(word_id, word_mask)  # 封装了文本输入和 mask
    bbox = torch.as_tensor(np.array(raw_batch[4]))  # 目标框信息，形状应为 [B, N, 4]
    batch = [img_data, text_data, bbox]
    return tuple(batch)  # 返回格式为：(img_data, text_data, bbox)，供模型 forward 使用


def collate_fn_clip(raw_batch):
    raw_batch = list(zip(*raw_batch))

    img = torch.stack(raw_batch[0])
    img_data = img
    text_data = list(raw_batch[1])
    bbox = torch.tensor(raw_batch[2])
    batch = [img_data, text_data, bbox]
    return tuple(batch)

#对一个二维列表中每一列分别求最大值，返回的是一个一维列表。
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """封装一个带掩码的张量。"""
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):#将 NestedTensor 中的 tensors 和 mask 一起转移到指定的设备（如 'cuda' 或 'cpu'）。
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):#返回原始的 tensors 和 mask。
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """将一个图像 Tensor 列表打包成一个统一尺寸的 Batch Tensor，并生成对应的 mask，
    用来构建 NestedTensor，便于后续模型处理（尤其是 Transformer 这类需要统一尺寸的模型）。"""
    # TODO make this more general
    if tensor_list[0].ndim == 3:#只支持 [C, H, W] 类型图像。否则报错。
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])#获取所有图像的 [C, H, W]，取每一维的最大值，保证能包住所有图像。
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size#这是 batch 的大小 B，即你有多少张图片。[B]+[C,H,W]=[B,C,H,W]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        #torch.zeros创建一个指定形状的张量（Tensor），并将其所有元素初始化为0
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)#[B,C,H,W],所有通道都共享一个 mask。
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)#创建一个 [B, H, W] 的全 1 mask，用于标记 padding 区域。
        for img, pad_img, m in zip(tensor_list, tensor, mask):#zip 将多个列表组合为元组列表一一对应
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)#把原图 img 拷贝到 pad_img 的左上角，其他区域（比如右下角）就留空（保持为 0）。
            m[: img.shape[1], :img.shape[2]] = False#把 mask 对应有效图像区域设置为 False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    作用是 在分布式训练中，只让主进程打印日志，其他进程不打印，避免重复输出。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查分布式训练是否可用并已初始化。"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """"#获得GPU数量"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """返回当前进程的编号（rank）"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """只让主进程保存模型或文件"""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """计算分类准确率
    output: 模型输出的 logits，形状为 [B, C]，表示每张图对每个类别的得分（logits）。
    target: 真实标签，形状为 [B]，每个值是一个类别的索引。
    topk: 是一个元组，例如 (1, 5) 表示计算 top-1 和 top-5 的准确率。
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)#取出 top-k 中最大的 k 值，如 (1, 5) 中的 5。
    batch_size = target.size(0)#== target.shape[0]获取当前 batch 的大小。
    #找到每个样本的前 maxk 个预测类别索引
    #topk 返回两个张量：最大值（得分），这里用 _ 忽略了，因为准确率计算只用到索引。
    #最大值对应的索引（类别编号），赋值给 pred。
    _, pred = output.topk(maxk, 1, True, True)#
    pred = pred.t()#转置，形状从 [B, maxk] 变为 [maxk, B]，便于接下来比较。
    # target.view(1, -1) 将 target 变为 [1, B] 的形状，expand_as(pred) 扩展为 [maxk, B]，与 pred 对齐。
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res#返回一个列表，比如当 topk=(1, 5) 时，res = [top1_accuracy, top5_accuracy]。


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    用于对张量进行插值（上采样/下采样），其功能类似于 torch.nn.functional.interpolate，
    但它兼容空 batch（即 batch size 为 0 的情况），并做了对旧版 PyTorch 的兼容处理。
    input          # 形状为 [B, C, H, W] 或 [B, C, D, H, W] 的张量
    size           # 目标输出大小，例如 [256, 256]
    scale_factor   # 缩放因子，例如 0.5、2.0
    mode           # 插值方式，如 "nearest"、"bilinear"、"trilinear"
    align_corners  # 对于 "bilinear" 或 "trilinear"，是否对齐角点
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
