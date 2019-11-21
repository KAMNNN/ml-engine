import squad
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import subprocess
from transformers import *

import spacy
from spacy.pipeline import EntityRecognizer
subprocess.call("python -m spacy download en_core_web_lg")
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_lg")
ner = EntityRecognizer(nlp.vocab)

def train(dataset):
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    gpt2 = GPT2Model.from_pretrained("gpt2-large")
    model = BertModel.from_pretrained('bert-large-uncased')
    
    for ipt in dataset:
        pass
       
def QuestionAnswer():
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    doc = nlp(text)
    ents = list(doc.ents)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    input_ids = tokenizer.encode("[CLS]") + tokenizer.encode(question) + tokenizer.encode("[SEP]") + tokenizer.encode(text) + tokenizer.encode("[SEP]")
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))



def main():
    QuestionAnswer()
    
    
if __name__ == "__main__":
    main()