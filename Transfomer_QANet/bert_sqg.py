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
    model = Bert_SQG(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    vocab_size = tokenizer.vocab_size
    #lm_labels = torch.LongTensor([0 for i in range(vocab_size)]).unsqueeze(0).to(device)
    for epoch in range(25):
        for IO in tqdm(dataset, desc='------ Training Epoch:{} ------'.format(epoch)):
            for i in range(0, len(IO)-1):
                model.zero_grad()
                outputs = model(IO[i])
                tmp = IO[i+1][:, :-1]
                loss = criterion(outputs.transpose(2, 1), tmp)
                loss.backward()
                optimizer.step()

        checkpoint = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, "./checkpoint/bert_sqg.pt")
        print()


def main():
    ds = data.BertSQG_DataClass()
    train(ds)


if __name__ == "__main__":
    main()
