import squad
import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import *
import data

import subprocess
import spacy
#subprocess.call("python -m spacy download en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm

context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."

def train(dataset):
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    gpt2 = GPT2Model.from_pretrained("gpt2-large")
    model = BertModel.from_pretrained('bert-large-uncased')
    
    for ipt in dataset:
        pass
   

def QuestionAnswer():
    doc = nlp(context)
    sents = [str(s.text) for s in doc.sents]
    #ents = list(doc.ents)
    #tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    #model = XLMForQuestionAnswering.from_pretrained('xlm-mlm-en-2048')
    #input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    #start_positions = torch.tensor([1])
    #end_positions = torch.tensor([4])
    #outputs = model(input_ids)
    #loss, start_scores, end_scores = outputs[0], outputs[1], outputs[2]
    #start, end = nn.Softmax(dim=1)(start_scores).argmax().item(), nn.Softmax(dim=1)(end_scores).argmax().item()
    #tmp = "Hello, my dog is cute"
    
    #print(start, end, tmp.split(' ')[start])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    input_ids = tokenizer.encode("[CLS]") + tokenizer.encode(sents[0]) + tokenizer.encode("[SEP]")
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    print()


def main():
    QuestionAnswer()
    
    
if __name__ == "__main__":
    train_data = data.data(True)
    t =  train_data[0]
    #main()