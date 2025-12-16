# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re
import torch
import codecs

UNK_TOKEN = '<unk>'## 未知词标记
PAD_TOKEN = '<pad>' # 填充标记（用于对齐句子长度）
END_TOKEN = '<eos>' # 句子结束标记
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # 按非单词字符分割句子


class Dictionary(object):
    """实现了一个简单的词典（Dictionary）类，用于管理单词和对应索引的映射，常用于自然语言处理（NLP）中的词表构建。"""
    def __init__(self):
        self.word2idx = {} # 存放单词->索引的字典
        self.idx2word = []# 存放索引->单词的列表

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]#返回单词对应的索引。

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a] # 通过索引查单词
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]# 批量索引查单词列表
        elif isinstance(a, str):
            return self.word2idx[a]  # 通过单词查索引
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx#判断某个单词是否已存在于词典中。


class Corpus(object):
    """用于加载文本、构建词典，并将句子转化为张量形式的词索引序列，常用于自然语言处理中的模型训练和推理前的预处理阶段。"""
    def __init__(self):
        self.dictionary = Dictionary()#创建一个 Dictionary 实例，用于存储词与索引的映射。

    def set_max_len(self, value):#设置最大句子长度（用于后续截断或填充）。
        self.max_len = value

    def load_file(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:#打开文件，逐行读取每个句子。
                line = line.strip()# 去除行首尾的空白字符。
                self.add_to_corpus(line)#对每行调用 add_to_corpus() 添加单词到词典中。
        self.dictionary.add_word(UNK_TOKEN)# 添加未知词标记到词典中。
        self.dictionary.add_word(PAD_TOKEN)# 添加填充标记到词典中。

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # 将一行句子按空格分割为单词，并转换为小写后逐个添加到词典中。
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=20):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())# 按非单词字符分割句子，得到单词列表。利用正则表达式将句子切分为词和标点（例如 hello! → ["hello", "!", ...]）。
        # words = [w.lower() for w in words if len(w) > 0]
        words = [w.lower() for w in words if (len(w) > 0 and w!=' ')]   # 去除空格，统一小写。

        if words[-1] == '.':#如果最后一个词是句点，则去掉它（避免干扰）
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                # words = [PAD_TOKEN] * (max_len - len(words)) + words
                words = words + [END_TOKEN] + [PAD_TOKEN] * (max_len - len(words) - 1)#如果句子短，就添加 <eos>（句末符）和 <pad> 进行填充。

        tokens = len(words) ## for end token
        ids = torch.LongTensor(tokens)#创建一个长整型张量来存储每个 token 对应的索引。
        token = 0
        for word in words:
            if word not in self.dictionary:#如果词不在词典中，替换成 <unk>。
                word = UNK_TOKEN
            # print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii','ignore').decode('ascii')))
            if type(word)!=type('a'):#如果类型不是字符串（某些特殊字符），则转码为 ASCII。
                print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii','ignore').decode('ascii')))
                word = word.encode('ascii','ignore').decode('ascii')
            ids[token] = self.dictionary[word]#查词典获取词的索引，并写入张量。
            token += 1
        # ids[token] = self.dictionary[END_TOKEN]
        return ids#返回张量格式的词索引列表。

    def __len__(self):
        return len(self.dictionary)#返回词典中的词数（包括普通词和特殊标记）
