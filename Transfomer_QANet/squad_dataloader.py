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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_letters = string.printable
nlp = spacy.load("en_core_web_sm")
ner_whitelist = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

class Squad_DataSet(Dataset):
    def __init__(self, datadir='./data', train=True):  
        TrainSet = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        TestSet = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

        self.data_dicts = {}
        self.all_ids_in_data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.token_ids = len(self.tokenizer.ids_to_tokens)
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        if(train):
            urllib.request.urlretrieve(TrainSet, os.path.join(datadir,'train-v2.0.json'))
            data = pd.read_json(os.path.join(datadir, 'train-v2.0.json'), orient='column')
            
        else:            
            urllib.request.urlretrieve(TestSet, os.path.join(datadir,'dev-v2.0.json'))
            data = pd.read_json(os.path.join(datadir, 'dev-v2.0.json'), orient='column')
           
        
        columns = ['title', 'text', 'context', 'question' 'answer', 'wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
        df = pd.DataFrame(columns=columns)

        def unicodeToAscii(s):
            return ''.join( c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)
        maxlen = 0
        
        for i in tqdm(range(len(data['data']))):
            for j in range(len(data['data'][i]['paragraphs'])):
                passage = unicodeToAscii(data['data'][i]['paragraphs'][j]['context'])

                passage_length = len(passage.split(" "))
                maxlen = max(maxlen, passage_length)

                while '\n' in passage:
                    index = passage.index('\n')
                    passage = passage[0:index] + passage[index + 1:]
                    
                for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):

                    question = unicodeToAscii(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                    id1 = data['data'][i]['paragraphs'][j]['qas'][k]['id']
                    
                    if()
                    answer_key = 'answers'

                    all_starts, all_ends, all_answers = [], [], []
                    for l in range(len(data['data'][i]['paragraphs'][j]['qas'][k][answer_key])):
                        answer = unicodeToAscii(data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['text'])
                        start = data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['answer_start']
                        end = data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['answer_start'] + len(answer)

                        while '\n' in answer:
                            index = answer.index('\n')
                            answer = answer[0:index] + answer[index + 1:]

                        all_starts.append(start)
                        all_ends.append(end)
                        all_answers.append(answer)

                    if len(set(all_starts)) != 1:
                        for popop in range(len(all_starts)):
                            if all_answers[popop] != '':
                                self.data_dicts['#'*popop + id1] = [passage, question, all_answers[popop], (all_starts[popop], all_ends[popop])]    
                                self.all_ids_in_data.append('#'*popop + id1)
                        
                    else:
                        self.data_dicts[id1] = [passage, question, all_answers[0], (all_starts[0], all_ends[0])]
                        self.all_ids_in_data.append(id1)
                    
        print(maxlen)
        assert (len(self.data_dicts) == len(self.all_ids_in_data))

    def __len__(self):
        return len(self.all_ids_in_data)
       
    def __getitem__(self, idx):

        id = self.all_ids_in_data[idx]
        data = self.data_dicts[id]
        context     = nlp(data[0])
        question    = nlp(data[1])
        answer      = nlp(data[2])
        start, end = data[3]
        

        ner_index = dict()
        def pos_ner_stop_punc(data, remove_stop=True):
            text = np.array([word.text for word in data], dtype=np.str)
            pos = np.array([word.pos_ for word in data], dtype=np.str)
            ner = np.array([(word.text, word.label_) for word in data.ents], dtype=np.str)
            is_stop = data.to_array(spacy.attrs.IS_STOP)
            is_punc = data.to_array(spacy.attrs.IS_PUNCT)
            shape = np.array([word.shape_ for word in data], dtype=np.str)
              
            counter = dict([(i, 0) for i in ner_whitelist])
            for x in range(len(ner)):
                if(ner[x][0] not in ner_index and ner[x][1] in ner_whitelist):
                    ner_index[ner[x][0]] = ner[x][1] + ' ' + str(counter[ner[x][1]])
                    counter[ner[x][1]] += 1
            if(remove_stop):     
                text = text[ (is_punc == 0) & (is_stop == 0) ]
            text = ' '.join(text)
                                
            for phrase, ner_obj in ner_index.items():
                text = text.replace(phrase, ner_obj)

            return text, pos, shape

        context_meta = pos_ner_stop_punc(context)
        question_meta = pos_ner_stop_punc(question, False)
        answer_meta = pos_ner_stop_punc(answer)

        ctx = context_meta[0].lower().split(' ')
        ctx_emb = [self.tokenizer.encode(' '.join(ctx[i:i + 512])) for i in range(0, 1024, 512)]
        ctx = list()
        for x in ctx_emb:
            ctx += x

        e1 = self.tokenizer.encode("[cls]") + ctx + self.tokenizer.encode("[seq]")
        e2 = self.tokenizer.encode("[cls]" + question_meta[0].lower() + "[seq]")
        e3 = self.tokenizer.encode("[cls]" + answer_meta[0].lower() + "[seq]")
        
        t1 = [e1[i] if(i < len(e1)) else 0 for i in range(1024) ] 
        t2 = [e2[i] if(i < len(e2)) else 0 for i in range(64) ] 
        t3 = [e3[i] if(i < len(e3)) else 0 for i in range(64) ] 
        
        return torch.tensor(t1, dtype=torch.int64, device=device),  \
               torch.tensor(t2, dtype=torch.int64, device=device),  \
               torch.tensor(t3, dtype=torch.int64, device=device)