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

#subprocess.call("python -m spacy download en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm

model = BERT_AGEN()
model = GPT2_QGEN()

def train(dataset):
    optimizer = AdamW(model.parameters(), lr = 3e-5)
    criterion = nn.CrossEntropyLoss()
    vocab_size = tokenizer.vocab_size
    lm_labels = torch.LongTensor([0 for i in range(vocab_size)]).unsqueeze(0)
    padding = tokenizer.encode("[PAD]")
    
    for epoch in range(25):
        for x,y,m in tqdm(dataset, desc='------ Training Epoch:{} ------'.format(epoch)):
            input_tensor = x
            output_tensor = x
            model.zero_grad()         
        
            for i in range(min(len(y), 32)):           
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
    ds = data.data()
    train(ds)
    
if __name__ == "__main__":
    main()