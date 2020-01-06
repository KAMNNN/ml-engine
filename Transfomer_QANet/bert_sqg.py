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

model = Bert_SQG()
device = data.device

def train(dataset):
    optimizer = AdamW(model.parameters(), lr = 3e-5)
    criterion = nn.CrossEntropyLoss()

    vocab_size = tokenizer.vocab_size
    #lm_labels = torch.LongTensor([0 for i in range(vocab_size)]).unsqueeze(0).to(device)
    padding = tokenizer.encode("[PAD]")
  
    for epoch in range(25):
        for x,y,m in tqdm(dataset, desc='------ Training Epoch:{} ------'.format(epoch)):
            input_tensor = x
            output_tensor = x
            model.zero_grad()         
        
            for i in range(min(len(y), 64)):           
                outputs = model(torch.cat([input_tensor, m], 1))
                input_tensor = outputs.argmax(dim=2)
                output_tensor = torch.cat([output_tensor, y[:,i].reshape(-1, 1)], 1)            
                loss = criterion(outputs.transpose(1,2), output_tensor)
                loss.backward()
                optimizer.step()

        checkpoint = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
        }            
        torch.save(checkpoint, "./checkpoint/bert_sqg.pt")
        print()

def main():
    ds = data.BertSQG_DataClass()
    train(ds)
    
if __name__ == "__main__":
    main()