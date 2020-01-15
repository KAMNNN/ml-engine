import os
import json
import torch
import urllib.request
import numpy as np
import torch.nn as nn
import subprocess
import spacy
from tqdm import tqdm
from collections import defaultdict
from transformers import *
from torch.utils.data import Dataset   
from multiprocessing import Pool

#subprocess.call("python -m spacy download en_core_web_lg")
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm
#subprocess.call("python -m spacy download en")

nlp = spacy.load("en_core_web_lg")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_SET = { 
    "squad" : "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "coqa" :  "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json",
    "quac" :  "https://s3.amazonaws.com/my89public/quac/train_v0.2.json"
}
DEV_SET = {
    "squad" : "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "coqa" :  "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json",
    "quac" :  "https://s3.amazonaws.com/my89public/quac/val_v0.2.json"
}

SQUAD_TRAIN = "./data/squad-train-v2.0.json"
SQUAD_DEV   = "./data/squad-dev-v2.0.json"
COQA_TRAIN = "./data/coqa-train-v1.0.json"
COQA_DEV   = "./data/coqa-dev-v1.0.json"
QUAC_TRAIN = "./data/quac-train-v0.2.json"
QUAC_DEV   = "./data/quac-dev-v0.2.json"

THREAD_NUM = 4

if not os.path.exists('./data'):
    os.makedirs('./data')
if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')

def Softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if not os.path.exists(SQUAD_TRAIN):
    urllib.request.urlretrieve(TRAIN_SET['squad'], SQUAD_TRAIN) 
if not os.path.exists(SQUAD_DEV):
    urllib.request.urlretrieve(DEV_SET['squad'], SQUAD_DEV)
if not os.path.exists(COQA_TRAIN):
    urllib.request.urlretrieve(TRAIN_SET['coqa'], COQA_TRAIN)
if not os.path.exists(COQA_DEV):
    urllib.request.urlretrieve(DEV_SET['coqa'], COQA_DEV)
if not os.path.exists(QUAC_TRAIN):
    urllib.request.urlretrieve(TRAIN_SET['quac'], QUAC_TRAIN)
if not os.path.exists(QUAC_DEV):
    urllib.request.urlretrieve(DEV_SET['quac'], QUAC_DEV)


class DataClass(Dataset): 
    def __init__ (self):
        self.data = list()       
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        pass

def _PreProcess():   
    s = [json.load(open(SQUAD_TRAIN)), json.load(open(SQUAD_DEV))]
    c = [json.load(open(COQA_TRAIN)), json.load(open(COQA_DEV))]
    q = [json.load(open(QUAC_TRAIN)), json.load(open(QUAC_DEV))]
    squad = defaultdict(list)
    coqa  = defaultdict(list)
    quac  = defaultdict(list)

    for d in s:
        for k, v in d.items():
            squad[k] += v
    for d in c:
        for k, v in d.items():
            coqa[k] += v
    for d in q:
        for k, v in d.items():
            quac[k] += v

    L1 = _preprocess_squad(squad['data'])
    #L2 = _preprocess_coqa(coqa['data'])
    #L3 = _preporcess_quac(quac['data'])

    return L1 #+ L2 #+ L3

def _get_answer_spans(para_text):
    para_nlp = nlp(para_text, disable=["tagger", "ner"])
    sentences = [(x.text, x.start_char) for x in para_nlp.sents]
    entities, entity_dict = [], {}

    sentence_lengths = [len(sentences[0][0])]
    for s in range(1, len(sentences)):
        sentence_lengths.append(sentence_lengths[s-1]+len(sentences[s][0]))        
    for x in para_nlp.ents:      
        if x.text in entity_dict:
            continue
        entity_dict[x.text] = 1
        for i in range(len(sentence_lengths)):
            if(x.start_char < sentence_lengths[i]):
                sent = sentences[i]
                break
            else:
                sent = sentences[0]
        entities.append((x.text, sent))
    return entities

def _neural_get_answer_spans(para_text, _processed_spans=[]):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if _processed_spans == []:
        _processed_spans = _get_answer_spans(para_text)
    context_doc = nlp(para_text)
    sentences = [str(sents) for sents in context_doc.sents]
    sent_lengths = [len(sentences[0])]
    for i in range(1, len(sentences)):
        sent_lengths.append(len(sentences[i]) + sent_lengths[i-1])
    spans = list()
    for span, start, end in _processed_spans:       
            for i in range(len(sent_lengths)):
                if(start < sent_lengths[i]):
                    sent = sentences[i]
                    break
                else:
                    sent = sentences[0]               
            spans.append(span.lower())
            input_ids = tokenizer.encode("[CLS]") + tokenizer.encode(sent) + tokenizer.encode("[SEP]")
            token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
            start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            bert_span = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).lower()
            if(bert_span not in spans and bert_span != ''):
                spans.append((bert_span, sent))     
    return spans

def _cleanup(x):
    context, question, answer, start = x

    def clean_text(text):
        text = text.replace("]", " ] ")
        text = text.replace("[", " [ ")
        text = text.replace("\n", " ")
        text = text.replace("''", '" ').replace("``", '" ')
        return text

    context_doc = nlp(clean_text(context), disable=["tagger", "ner"])
    #question_doc = nlp(clean_text(question), disable=["tagger", "ner"])
    #answer_doc = nlp(clean_text(answer), disable=["tagger", "ner"])
    
    output_context  = defaultdict(list)
    output_question = defaultdict(list)
    output_answer =   defaultdict(list)

    sentences = [str(sents) for sents in context.split('.')]
    sent_lengths = [len(sentences[0])]
    for i in range(1, len(sentences)):
        sent_lengths.append(len(sentences[i]) + sent_lengths[i-1])
    for i in range(len(sent_lengths)):
                if(start < sent_lengths[i]):
                    sent = sentences[i]
                    break
                else:
                    sent = sentences[0]
       
    output_context['text'] =    clean_text(context)
    output_question['text'] =   clean_text(question)
    output_answer['text'] =     clean_text(answer)

    #spans = _get_answer_spans(clean_text(context)) 
    #spans = _neural_get_answer_spans(clean_text(context), spans) 

    output = {
        "text": {  
            "context": output_context['text'],
            "question": output_question['text'],
            "answer": output_answer['text'],
        },
        "sentence": sent,
        
    }

    return output

def _preprocess_squad(data):   
    tmp = list()
    p = Pool(THREAD_NUM)
    count = 0
    for entry in tqdm(data, desc='PreProcess Squad'):
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question = qa["question"]
                is_impossible = qa["is_impossible"]                    
                if not is_impossible:
                    answer =            qa["answers"][0]
                    orig_answer_text =  answer["text"]
                    start =             answer["answer_start"]
                    if(count >= 8):
                        break
                    if(answer != '' and question != '' and orig_answer_text != ''):
                        tmp.append((context, question, orig_answer_text, start))
                        count += 1
                    

    output = list(tqdm(p.imap(_cleanup, tmp), total=len(tmp), desc='squad cleanup'))   
    return output

def _preprocess_coqa(data):
    output = list()
    tmp = list()
    p = Pool(THREAD_NUM)
    for entry in tqdm(data, desc='PreProcessing Coqa'):
        context = entry['story']
        for q, a in zip(entry['questions'], entry['answers']):
            question = q['input_text']
            answer = a['input_text']
            answer_text = a['span_text']
            start = a['span_start']
            if(answer == ''):
                answer = context[ a['span_start']:a['span_end']]    
            tmp.append((context, question, answer_text, start))

    output = list(tqdm(p.imap(_cleanup, tmp), total=len(tmp), desc='coqa cleanup'))   

    return output

def _preporcess_quac(data):
    output = list()
    tmp = list()
    p = Pool(THREAD_NUM)
    for entry in tqdm(data, desc='PreProcessing Quac'):
        background = entry['background']
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']                
                for a in qa['answers']:
                    answer = a['text']
                    start = a['answer_start']
                    followup = qa['followup']
                    yesno = qa['yesno']
                    tmp.append((context, question, answer, start))

    output = list(tqdm(p.imap(_cleanup, tmp), total=len(tmp), desc='quac cleanup'))
    return output


class BertSQG_DataClass(DataClass):
    def __init__(self, max_size=512):        
        super(BertSQG_DataClass, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = _PreProcess() 
        self.max_size = max_size
        self.training_data = [self.init_helper(x) for x in tqdm(data, total=len(data), desc='Setting up training data')]
        
    def init_helper(self, d):
        c,q,a,s = d['text']['context'], d['text']['question'], d['text']['answer'], d['sentence']
        input_str = "[CLS]" + s + "[SEP]" + a + "[SEP]" 
        input_tokens = self.tokenizer.encode(input_str)
        mask_tokens = self.tokenizer.encode("[MASK]")  
        output_tokens = self.tokenizer.encode(q)        
        output = [input_tokens + mask_tokens]
        for i in range(len(output_tokens)):
            tokens = output[-1]
            tokens[-1] = output_tokens[i]
            output.append(tokens+mask_tokens)
        return output

    def __len__(self):
        return len(self.training_data)
    def __getitem__(self, idx):
        output = list()
        for x in self.training_data[idx]:
            tensor = torch.zeros(self.max_size)
            for i in range(len(x)):
                tensor[i] = x[i]
            output.append(tensor)
        return output

        
        
class Bert_GPT2_DataClass(DataClass):
    def __init__(self):
        super(Bert_GPT2_DataClass, self).__init__()
