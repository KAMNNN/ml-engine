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

context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
Q1 = "In what country is Normandy located?"
A1 = "France"
Q2 = "When were the Normans in Normandy?"
A2 = "10th and 11th centuries"
Q3 = "From which countries did the Norse originate?"
A3 = "Denmark"

model = Bert_QGen()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
softmax = nn.Softmax(dim=0)

def train(dataset):
    optimizer = AdamW(model.parameters(), lr = 3e-5)
    criterion = nn.CrossEntropyLoss()
    vocab_size = tokenizer.vocab_size
    lm_labels = torch.LongTensor([0 for i in range(vocab_size)]).unsqueeze(0)
    padding = tokenizer.encode("[PAD]")
    
    #input = torch.randn(3, 5, requires_grad=True) #5, 3
    #target = torch.empty(3, dtype=torch.long).random_(5) #3
    #output = criterion(input, target)


    for x,y,m in tqdm(dataset, desc='------ Training ------'):
        input_tensor = x
        output_tensor = x
        model.zero_grad()         
       
        for i in range(5):           
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
    torch.save(checkpoint, "bert_sqg.pt")
    print()

def eval(dataset):
    for c,q,a,s in tqdm(dataset, desc='------ Evaluation ------'):
       pass 

 

def main():
    ds = data.data()
    train(ds)
    
if __name__ == "__main__":
    #from transformers import GPT2LMHeadModel, GPT2Tokenizer
    #import torch

    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #model = GPT2LMHeadModel.from_pretrained('gpt2')

    #generated = tokenizer.encode("The Manhattan bridge")
    #context = torch.tensor([generated])
    #past = None

    #for i in range(100):
    #    output, past = model(context, past=past)
    #    token = torch.argmax(output[0, :])
    #    generated += [token.tolist()]
    #    context = token.unsqueeze(0)

    #sequence = tokenizer.decode(generated)
    
    main()