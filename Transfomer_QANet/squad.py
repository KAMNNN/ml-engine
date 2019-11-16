import os
import urllib.request
import unicodedata
import string
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import *
from nltk.translate import bleu_score

import spacy
import torch

from squad_utils import *

whitespace = tokenizer.whitespace_tokenize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_letters = string.printable
nlp = spacy.load("en_core_web_sm")
ner_whitelist = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']


class SquadObj(object):
   
    def __init__(self, qas_id, question_text, doc_tokens, orig_answer_text=None, start_position=None, end_position=None, is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class Squad(Dataset):
    def __init__(self, datadir='./data', train=True):  
        TrainSet = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        TestSet = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        self.data_lst = []        
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.token_ids = len(self.tokenizer.ids_to_tokens)
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        if(train):
            urllib.request.urlretrieve(TrainSet, os.path.join(datadir,'train-v2.0.json'))
            data = pd.read_json(os.path.join(datadir, 'train-v2.0.json'), orient='column')['data']          
        else:            
            urllib.request.urlretrieve(TestSet, os.path.join(datadir,'dev-v2.0.json'))
            data = pd.read_json(os.path.join(datadir, 'dev-v2.0.json'), orient='column')['data']
        
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        entrys = list()

        for entry in data:
            for paragraph in data['paragraphs']:
                passage = paragraph['context']
                doc_tokens, char_to_word_offset, prev_is_whitespace = [], [], False
                for c in passage:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)
                for qa in passage['qas']:
                    qas_id, question_text = qa['id'], qa['question']
                    start_pos, end_pos, orig_answer_text, is_impossible = None, None, None, False
                    if train:
                        answer = qa['answers'][0]
                        original_answer_text = qa['answers'][0]['text']
                        answer_offset = answer['answer_start']
                        answer_length = len(original_answer_text)
                        start_pos = char_to_word_offset[answer_offset]
                        end_pos = char_to_word_offset[answer_offset + answer_length - 1]
                        actual_text = " ".join(doc_tokens[start_pos:end_pos+1])                        
                        cleaned_answer_text = " ".join(whitespace(original_answer_text))
                    else:
                        start_pos = -1
                        end_pos = -1
                        original_answer_text = ""
                    entrys.append(SquadObj(qas_ids, question_text, doc_tokens, orig_answer_text, start_pos, end_pos, is_impossible)) 
        
        self.data_lst = convert_obj_to_features(entrys, self.tokenizer, 384, 128, 64, train)     
        
    def __len__(self):
        return len(self.data_lst)
       
    def __getitem__(self, idx):
       return self.data_lst[idx]



