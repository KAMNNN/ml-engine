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

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")

def train(dataset):
    model1 = BERT_AGEN().to(device)
    model2 = GPT2_QGEN().to(device)

    optimizer1 = AdamW(model1.parameters(), lr = 3e-5)
    optimizer2 = AdamW(model2.parameters(), lr = 3e-5)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()


    for epoch in range(25):
        for x,y,m in tqdm(dataset, desc='------ Training Epoch:{} ------'.format(epoch)):
            input_tensor = x
            output_tensor = x
            
            question_Generated = model1()
            
        

            answer_Generated = model2(question_Generated)
            
            
            optimizer1.zero_grad()        
            optimizer2.zero_grad()      

        checkpoint = {
            'model': model1,
            'state_dict': model1.state_dict(),
            'optimizer' : optimizer.state_dict()
        }            
        torch.save(checkpoint, "./checkpoint/bert_ag.pt")
        checkpoint = {
            'model': model2,
            'state_dict': model2.state_dict(),
            'optimizer' : optimizer.state_dict()
        }            
        torch.save(checkpoint, "./checkpoint/bert_ag.pt")

        print()

def main():
    ds = data.data()
    dl = DataLoader(ds, num_workers=4, batch_sampler=2)
    train(ds)
    
if __name__ == "__main__":
    main()
