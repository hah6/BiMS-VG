import torch
from torch import nn
from utils.misc import NestedTensor
from transformers import BertModel
class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, enc_num: int):
        """
        name: 预训练 BERT 模型名称，如 'bert-base-uncased'
        train_bert: 是否训练 BERT 的参数
        enc_num: 如果 >0，表示提取第 enc_num 层 encoder 输出；如果为0，提取词嵌入（不经过 encoder）
        """
        super().__init__()
        self.num_channels = 768 if name == 'bert-base-uncased' else 1024
        self.enc_num = enc_num

        # 加载 HuggingFace BERT 模型，并启用输出所有中间层
        self.bert = BertModel.from_pretrained(name, output_hidden_states=True)

        # 冻结 BERT 参数（如果不希望训练）
        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        """
        tensor_list: NestedTensor，包含:
            - tensor_list.tensors: [B, T] 的 token ids
            - tensor_list.mask:    [B, T] 的 attention mask（1 表示有效 token）
        """
        outputs = self.bert(
            input_ids=tensor_list.tensors,
            attention_mask=tensor_list.mask,
            token_type_ids=None
        )

        if self.enc_num > 0:
            # hidden_states[0] 是 embedding，之后是每一层 encoder 的输出
            hidden_states = outputs.hidden_states
            xs = hidden_states[self.enc_num-1]
        else:
            # 仅提取 embedding（未经过 encoder）
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        # 构造输出的 mask：True 表示 padding，False 表示有效 token
        mask = ~tensor_list.mask.to(torch.bool)
        return NestedTensor(xs, mask)
def build_bert(args):
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.bert_enc_num)
    return bert