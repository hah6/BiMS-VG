import sys
import re
from torch.utils.data import Dataset
import os.path as osp
import torch
import numpy as np
from PIL import Image
from tensorboard.plugins.text.text_plugin import text_array_to_html
from transformers import BertTokenizer
from utils.word_utils import Corpus

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

sys.path.append('.')
def read_examples(input_line,unique_id):
    """输入：单行文本 (input_line) 和唯一标识符 (unique_id)

输出：包含一个或多个 InputExample 对象的列表

核心逻辑：用正则表达式分割文本，构造结构化数据对象"""

    examples = []
    line = input_line.strip()
    text_a=None
    text_b=None

    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a=line
    else:
        text_a=m.group(1)
        text_b=m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples
class InputFeatures(object):
    #这个类表示一组模型输入特征
    def __init__(self, unique_id, tokens,input_ids, input_mask, segment_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids   #每个 token 对应的词表 ID，供模型使用
        self.input_mask = input_mask #mask 向量（通常为 1 的部分表示实际输入，0 为 padding）
        self.segment_ids = segment_ids#段 ID（用于区分句子对中的两个句子，通常是 0 和 1）
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """就地截断两个 token 序列，使它们的总长度不超过给定的最大长度 max_length。"""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):  #.pop() 默认是移除列表最后一个元素（也就是末尾的 token）。
            tokens_a.pop()
        else:
            tokens_b.pop()
def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """将输入示例转换为模型输入特征。

    参数:
        examples: 输入示例列表。
        tokenizer: 用于分词的 tokenizer。
        max_seq_length: 最大序列长度。

    返回:
        特征列表，每个特征包含唯一标识符、输入 ID、输入掩码和段 ID。
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)#对句子 text_a 进行分词。
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)#3个特殊标记（[CLS]、[SEP]、[SEP]）的长度。
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]#切片（slice）操作,取 tokens_a 列表从开头（索引 0）开始，到索引 (max_seq_length - 2) - 1 位置结束的所有元素。

        # 添加特殊标记
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)#Python 中生成列表的简洁写法，创建一个长度和 tokens 一样长、所有元素都为 0 的列表

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)#. 将tokens转换为对应词表中的id
        input_mask = [1] * len(input_ids)

        # 填充到最大长度
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(unique_id=example.unique_id,
                                      tokens=tokens,
                                      input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids))
    return features
class DatasetNotFoundError(Exception):
    """自定义异常类，用于处理数据集未找到的情况。"""
    def __init__(self, message):
        super().__init__(message)
class GroundingDataset(Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')
        }
    }
    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False,
                 bert_model='bert-base-uncased'):
        super().__init__()
        self.data_root = data_root
        self.split_root = split_root#存放数据划分文件（如 train.pth, val.pth）的目录。
        self.dataset = dataset#数据集名称
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split#当前加载的数据划分，如 'train', 'val', 'testA', 'testB' 等。
        self.return_idx = return_idx
        # 使用预训练的 BERT tokenizer，do_lower_case=True 表示将所有字母转换为小写，这通常用于 uncased 模型
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.images = []
        if self.transform is  None:
            raise ValueError("Transform must be provided for the dataset.")
        if split=='train':
            self.augment = True
        else:
            self.augment = False
 # 根据数据集类型设置图像和 split 路径
        if dataset == 'referit':
            self.dataset_root = osp.join(data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif dataset == 'flickr':
            self.dataset_root = osp.join(data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        else:
            self.dataset_root = data_root
            self.im_dir = osp.join(self.dataset_root, 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        if not self.exists_dataset():
            raise DatasetNotFoundError(f"Dataset '{self.dataset}' not found in '{self.split_root}'.")
            # 检查 split 是否合法
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']
        if split not in valid_splits:
            raise ValueError(f"数据集 {self.dataset} 不支持 split: {split}")


        dataset_path = osp.join(self.split_root, self.dataset)
        # 处理 split 文件
        splits = ['train', 'val'] if split == 'trainval' and dataset != 'referit' else [split]
        for s in splits:
            # 加载该划分的索引数据（通常是一个列表，包含图像文件名、边界框、文本等）
            split_file = osp.join(dataset_path, f'{dataset}_{s}.pth')
            # 将加载的数据合并到 self.images 列表中，方便后续数据读取
            self.images += torch.load(split_file)

        # LSTM 模式下加载语料
        if self.lstm:
            self.corpus = torch.load(osp.join(dataset_path, 'corpus.pth'))


    def exists_dataset(self):
        """判断数据集 split 路径是否存在"""
        return osp.exists(osp.join(self.split_root, self.dataset))

    def tokenize_phrase(self, phrase):
        """将文本短语转换为词 ID 序列（用于 LSTM 模式）"""
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        """将词 ID 反转为词语（用于可视化）"""
        return self.corpus.dictionary[words]
    def pull_item(self, idx):
        """根据索引提取一条图像-文本-框样本"""

        # 对于 flickr30k 数据集，图像元组是 (img_file, bbox, phrase)
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            # 对于 referit/refcoco 等数据集，图像元组是 (img_file, sent_id, bbox, phrase, image_size)
            img_file, _, bbox, phrase, _ = self.images[idx]

        bbox = np.array(bbox, dtype=int)

        # 对于 COCO 类型数据集（不是 referit/flickr），bbox 是 xywh 格式，需要转换为 xyxy
        if self.dataset not in ['referit', 'flickr']:
            bbox[2] += bbox[0]  # bbox[2] 是宽度，加上左上角 x 得到右下角 x
            bbox[3] += bbox[1]  # bbox[3] 是高度，加上左上角 y 得到右下角 y

        # 拼接图像路径并读取图像为 RGB 格式
        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        # 返回图像，转为小写的短语，以及浮点型 bounding box
        return img, phrase.lower(), torch.tensor(bbox, dtype=torch.float32)
    def __getitem__(self, idx):
        """获取一条样本，返回图像、掩码、词 ID、mask 和目标框"""
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        # 应用图像增强 transform
        input_dict = self.transform({'img': img, 'box': bbox, 'text': phrase})
        img, bbox, phrase, img_mask = input_dict['img'], input_dict['box'], input_dict['text'], input_dict['mask']

        # 文本编码
        if self.lstm:
            word_id = self.tokenize_phrase(phrase)
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, tokenizer=self.tokenizer,max_seq_length=self.query_len)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        # 测试模式返回附加变量（缩放比例、偏移等）
        if self.testmode:
            ratio, dw, dh = np.array(1.0, dtype=np.float32), np.array(0.0), np.array(0.0)
            return img, np.array(word_id), np.array(word_mask), bbox.numpy(), ratio, dw, dh, self.images[idx][0]

        return img, np.array(img_mask), np.array(word_id,dtype=int), np.array(word_mask,dtype=int), np.array(bbox ,dtype=np.float32)
    def __len__(self):
        # return int(len(self.images) / 10)
        return len(self.images)