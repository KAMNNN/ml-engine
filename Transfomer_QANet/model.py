import torch
import torch.nn as nn
from transformers import *

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class Bert_QGen(nn.Module):
    def __init__(self):
        super(Bert_QGen, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.train()
        self.cls = nn.Linear(self.bert.config.hidden_size, tokenizer.vocab_size)
        self.bias = torch.zeros(tokenizer.vocab_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        h = self.bert(x)
        h = self.softmax(self.cls(h[0]) + self.bias)

        return h