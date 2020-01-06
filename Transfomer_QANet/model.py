import torch
import torch.nn as nn
from transformers import *

BERT_TYPE = "bert-base-uncased"
GPT2_TYPE = "gpt2"
tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)
tokenizer2 = GPT2Tokenizer.from_pretrained(GPT2_TYPE)

class Bert_SQG(nn.Module):
    def __init__(self):
        super(Bert_QGen, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_TYPE)
        self.bert.train()
        self.cls = nn.Linear(self.bert.config.hidden_size, tokenizer.vocab_size)
        self.bias = torch.zeros(tokenizer.vocab_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        h = self.bert(x)
        h = self.softmax(self.cls(h[0]) + self.bias)
        return h

class GPT2_QGEN(nn.Module):
    def __init__(self):
        super(GPT2_QGEN, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(GPT2_TYPE)
        self.gpt2.train()

    def forward(self, x):
        h = self.gpt2(x)
        pass

class BERT_AGEN(nn.Module):
    def __init__(self):
        super(BERT_AGEN, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_TYPE)
        self.bert.train()

    def forward(self, x):
        h = self.bert(x)
        pass


