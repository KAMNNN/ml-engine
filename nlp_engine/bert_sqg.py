import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import *
from tqdm import tqdm
from model import *
import subprocess
import spacy
import data

device = data.device

def train(dataset):
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
        print()


def main():
    ds = data.BertSQG_DataClass()
    dl = DataLoader(ds, num_workers=4, batch_size=4)
    train(dl)


if __name__ == "__main__":
    main()