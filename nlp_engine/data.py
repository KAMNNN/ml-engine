import os
import six
import json
import torch
import urllib.request
import numpy as np
import torch.nn as nn
import subprocess
import spacy
import zipfile
from tqdm import tqdm
from collections import defaultdict
from transformers import *
from torch.utils.data import Dataset   
from multiprocessing import Pool
import multiprocessing as mp


from gensim.corpora import WikiCorpus
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.wrappers import FastText

nlp = spacy.load("en")
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

GLOVE_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_DATA = './data/glove.840B.300d.txt'
GLOVE_VEC = './data/word2vec-glove.840B.300d.txt'

WIKI_URL = "http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
WIKI_DATA = "./data/enwiki-latest-pages-articles.xml.bz2"
WIKI_EXTRACT_DATA = "./data/wiki.en.text"
WIKI_MODEL = "./data/wiki.en.word2vec.model"

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip"
FASTTEXT_BIN = "./data/wiki.en.bin"

SQUAD_TRAIN = "./data/squad-train-v2.0.json"
SQUAD_DEV   = "./data/squad-dev-v2.0.json"
COQA_TRAIN = "./data/coqa-train-v1.0.json"
COQA_DEV   = "./data/coqa-dev-v1.0.json"
QUAC_TRAIN = "./data/quac-train-v0.2.json"
QUAC_DEV   = "./data/quac-dev-v0.2.json"

THREAD_NUM = mp.cpu_count() // 2

if not os.path.exists('./data'):
    os.makedirs('./data')
    if not os.path.exists('./data/wiki'):
        os.makedirs('./data/wiki')
if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')
if not os.path.exists('./logs'):
    os.makedirs('./logs')
    
def vectorize(wikipedia=False, fasttext=False):
    if(fasttext):
        model = FastText.load_fasttext_format('wiki.en.bin')
        return model
    elif(wikipedia):
        if not os.path.exists('./data/wiki_word_vec.kv'):
            wiki = WikiCorpus(WIKI_DATA, lemmatize=False, dictionary={})
            sentences = list(wiki.get_texts())
            model = Word2Vec(sg=1, hs=1, size=300, sample=1e-3, iter=5, min_count=10)
            model.init_sims(replace=True)
            model.build_vocab(sentences)
            model.train(sentences=sentences, total_examples=len(sentences), epochs=10)
            word_vectors = model.wv
            word_vectors.save_word2vec_format('./data/wiki_word_vec.bin')
            return word_vectors
        else:
            return Word2Vec.load('./data/wiki_word_vec.bin')           
    else:
        if not os.path.exists('./data/glove_word_vec.kv'):
            glove2word2vec(GLOVE_DATA, GLOVE_VEC)
            model = KeyedVectors.load_word2vec_format(GLOVE_VEC)
            word_vectors = model.wv
            word_vectors.save_word2vec_format('./data/glove_word_vec.bin')
            return word_vectors
        else:
            return Word2Vec.load("./data/glove_word_vec.bin")
    
def Softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if not os.path.exists(GLOVE_DATA):
    urllib.request.urlretrieve(GLOVE_URL, './data/glove.840B.300d.zip')
    with zipfile.ZipFile('./data/glove.840B.300d.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')
    os.remove('./data/glove.840B.300d.zip')
if not os.path.exists(FASTTEXT_DATA):
    urllib.request.urlretrieve(FASTTEXT_URL, './data/wiki.en.zip')
    with zipfile.ZipFile('./data/wiki.en.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')
    os.remove('./data/wiki.en.zip')

if not os.path.exists(WIKI_DATA):
    urllib.request.urlretrieve(WIKI_URL, WIKI_DATA)
if not os.path.exists(SQUAD_TRAIN):
    urllib.request.urlretrieve(TRAIN_SET['squad'], SQUAD_TRAIN) 
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
        self.contexts, self.questions, self.answers = _PreProcess(True, False, False)
    def __len__(self):
        return len(self.answers)
    def __getitem__(self, idx):
        pass

def _PreProcess(squad=True, coqa=True, quac=True):   
    s = [json.load(open(SQUAD_TRAIN)), json.load(open(SQUAD_DEV))]
    c = [json.load(open(COQA_TRAIN)), json.load(open(COQA_DEV))]
    q = [json.load(open(QUAC_TRAIN)), json.load(open(QUAC_DEV))]
    SQUAD = defaultdict(list)
    COQA  = defaultdict(list)
    QUAC  = defaultdict(list)

    for d in s:
        for k, v in d.items():
            SQUAD[k] += v
    for d in c:
        for k, v in d.items():
            COQA[k] += v
    for d in q:
        for k, v in d.items():
            QUAC[k] += v

    ctx = list()
    que = list()
    ans = list()

    context = -1
    question = -1
    if(squad):
        for entry in tqdm(SQUAD['data'], desc='PreProcess Squad'):
            for paragraph in entry["paragraphs"]:
                ctx.append(paragraph['context'])
                context += 1
                for qa in paragraph["qas"]:
                    que.append(qa['question'])
                    question += 1
                    if not qa["is_impossible"]:
                        for answer in qa['answers']:
                            ans.append([answer['text'], answer["answer_start"], context, question])

    if(coqa): 
        for entry in tqdm(COQA['data'], desc='PreProcessing Coqa'):
            ctx.append(entry['story'])
            context += 1
            for q,a in zip(entry['questions'], entry['answers']):
                que.append(q['input_text'])
                question += 1
                if(a['input_text'] == ''):
                    ans.append(ctx[context][a['span_start']:a['span_end'], a['span_start'], context, question])
                else:
                    ans.append([a['input_text'], a['span_start'], context, question])
    if(quac):
        for entry in tqdm(QUAC['data'], desc='PreProcessing Quac'):
            for paragraph in entry['paragraphs']:
                ctx.append(paragraph['context'])
                context += 1
                for qa in paragraph['qas']:
                    que.append(qa['question'])
                    question += 1
                    for a in qa['answers']:                        
                        followup = qa['followup']
                        yesno = qa['yesno']
                        ans.append([a['answer'], a['answer_start'], context, question])
    return (ctx, que, ans)

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

class BertSQG_DataClass(DataClass):
    def __init__(self, max_size=512):        
        super(BertSQG_DataClass, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_size = max_size
        
    
    def _setup(self, context, question, answer, start):
        def clean_text(text):
            text = text.replace("]", " ] ")
            text = text.replace("[", " [ ")
            text = text.replace("\n", " ")
            text = text.replace("''", '" ').replace("``", '" ')
            return text

        context_doc = nlp(clean_text(context), disable=["tagger", "ner"])           
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
       
        #spans = _get_answer_spans(clean_text(context)) 
        #spans = _neural_get_answer_spans(clean_text(context), spans) 

        return clean_text(context),  clean_text(question), clean_text(answer), sent

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        answer, start, context, question = self.answers[idx]
        c,q,a,s = self._setup(self.contexts[context], self.questions[question], answer, start)        
        input_str = "[CLS]" + s + "[SEP]" + a + "[SEP]" 
        input_tokens = self.tokenizer.encode(input_str)
        mask_tokens = self.tokenizer.encode("[MASK]")  
        output_tokens = self.tokenizer.encode(q)        
        div_tokens = [input_tokens + mask_tokens]
        for i in range(len(output_tokens)):
            tokens = div_tokens[-1]
            tokens[-1] = output_tokens[i]
            div_tokens.append(tokens+mask_tokens)

        output = list()
        for x in div_tokens:
            tensor = torch.zeros(self.max_size, dtype=torch.long)
            for i in range(len(x)):
                tensor[i] = x[i]
            output.append(tensor)
        return output

        
        
class Bert_GPT2_DataClass(DataClass):
    def __init__(self):
        super(Bert_GPT2_DataClass, self).__init__()
        self.tokenizer1 = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer2 = GPT2Tokenizer.from_pretrained("gpt2")
        self.SPECIAL_TOKENS = [ "<bos>", "<eos>", "<paragraph>", "<answer-general>", "<answer-specific>", "<question-general>", "<question-specific>", "<pad>" ]
        self.MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
        
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        bos, eos, paragraph, answer_general, answer_specific, question_general, question_specific = self.tokenizer2.convert_tokens_to_ids(self.SPECIAL_TOKENS[:-1])
   
        answer, start, context_id, question_id = self.answers[idx]
        context_tokens = self.tokenizer2(self.contexts[context_id])
        question_tokens = self.tokenizer2(self.questions[question_id])
        answer_tokens = self.tokenizer2(answer)
        

        sequence = [bos] + context_tokens
        token_types = [ answer_general if ((i - 1) >= token_start and (i - 1) < token_end) else paragraph  for i in range(len(context_tokens) + 1)]
        lm_labels = [-1 for _ in range(len(context_tokens)+1)]
        
        sequence.extends([answer_general] + answer_tokens)
        token_types.extend([answer_general for _ in range(len(answer_tokens) + 2)])
        lm_labels.extend([-1 for _ in range(len(answer_tokens) + 1)])

        sequence.extends([question_general] + question_tokens + [eos])
        token_types.extend([question_general for _ in range(len(question_tokens) + 2)])        
        lm_labels.extend([-1] + question_tokens + [eos])
        assert len(sequence) == len(token_types)
        assert len(token_types) == len(lm_labels)
        instance = {
            "input_ids": torch.tensor(sequence).to(device),
            "token_type_ids": torch.tensor(token_types).to(device),
            "lm_labels": torch.tensor(lm_labels).to(device)
        }

        padding = self.tokenizer2.convert_tokens_to_ids(self.SPECIAL_TOKENS[-1])
        max_l = self.tokenizer2.max_len()
        for name in instance.keys():     
            instance[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in instance[name]]

     

        return instance 