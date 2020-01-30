import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import *
from tqdm import tqdm
import spacy
import data


BERT_TYPE = "bert-base-uncased"
device = data.device
tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)
class Bert_SQG(nn.Module):
    def __init__(self):
        super(Bert_SQG, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_TYPE)
        self.bert.train()
        self.cls = nn.Linear(self.bert.config.hidden_size, tokenizer.vocab_size)
        self.bias = torch.zeros(tokenizer.vocab_size)
        self.softmax = nn.Softmax(dim=1)
        #if(torch.cuda.device_count() > 1):
        #    self.bert = nn.DataParallel(self.bert)

    def forward(self, x):
        h = self.bert(x)
        h = self.softmax(self.cls(h[0]) + self.bias)
        return h
    
def train():
    ds = data.BertSQG_DataClass()
    dataset = DataLoader(ds, num_workers=4, batch_size=4)
    model = Bert_SQG()
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    vocab_size = tokenizer.vocab_size
    
    for epoch in range(25):
        with tqdm(total=len(dataset), desc='------ Training Epoch:{} ------'.format(epoch)) as t:
            for _id, IO in enumerate(dataset):
                for i in range(0, len(IO)-1):
                    ipt = IO[i].to(device)
                    outputs = model(ipt)
                    tmp = IO[i+1].to(device)
                    tmp2 = outputs.transpose(2,1)
                    loss = criterion(tmp2, tmp)
                    if(torch.cuda.device_count() > 1):
                        loss = loss.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                t.set_postfix(loss='{:05.3f}'.format(loss.item()))
                t.update()
        checkpoint = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, "./checkpoint/bert_sqg.pt")
        print('Trained Model')
