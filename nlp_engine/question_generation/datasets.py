import os
import json
import zipfile
from tqdm.auto import tqdm
import urllib.request
import tensorflow_datasets as tfds

from gensim.corpora import WikiCorpus
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.wrappers import FastText

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


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

if not os.path.exists('./data'):
    os.makedirs('./data')
if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')
if not os.path.exists('./logs'):
    os.makedirs('./logs')
    

def vectorize(wikipedia=False, fasttext=False):
    if(fasttext):
        if not os.path.exists(FASTTEXT_BIN):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=FASTTEXT_URL.split('/')[-1]) as t:
                urllib.request.urlretrieve(FASTTEXT_URL, './data/wiki.en.zip', reporthook=t.update_to)
            with zipfile.ZipFile('./data/wiki.en.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/')
            os.remove('./data/wiki.en.zip')    
       
            model = FastText.load_fasttext_format(FASTTEXT_BIN)
            return model
 
    elif(wikipedia):
        if not os.path.exists(WIKI_DATA):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=WIKI_URL.split('/')[-1]) as t:    
                urllib.request.urlretrieve(WIKI_URL, WIKI_DATA)
        if not os.path.exists('./data/wiki_word_vec.bin'):
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
        if not os.path.exists(GLOVE_DATA):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=GLOVE_URL.split('/')[-1]) as t:
                urllib.request.urlretrieve(GLOVE_URL, './data/glove.840B.300d.zip', reporthook=t.update_to)
            with zipfile.ZipFile('./data/glove.840B.300d.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/')
            os.remove('./data/glove.840B.300d.zip')

        if not os.path.exists('./data/glove_word_vec.bin'):
            glove2word2vec(GLOVE_DATA, './data/word2vec-glove.840B.300d.txt')
            model = KeyedVectors.load_word2vec_format('./data/word2vec-glove.840B.300d.txt')
            word_vectors = model.wv
            word_vectors.save_word2vec_format('./data/glove_word_vec.bin')
            return word_vectors
        else:
            return Word2Vec.load("./data/glove_word_vec.bin")


def squad(dev=False):
    ctx,que,ans = list(), list(), list()
    context, question = -1, -1
    if not os.path.exists(SQUAD_TRAIN):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=TRAIN_SET['squad'].split('/')[-1]) as t:    
            urllib.request.urlretrieve(TRAIN_SET['squad'], SQUAD_TRAIN) 
    if not os.path.exists(SQUAD_DEV):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=DEV_SET['squad'].split('/')[-1]) as t:    
            urllib.request.urlretrieve(DEV_SET['squad'], SQUAD_DEV)
    if dev:
        data = json.load(open(SQUAD_DEV))
    else:
        data = json.load(open(SQUAD_TRAIN))

    for entry in tqdm(data['data'], desc='PreProcess Squad'):
        for paragraph in entry["paragraphs"]:
            ctx.append(paragraph['context'])
            context += 1
            for qa in paragraph["qas"]:
                que.append(qa['question'])
                question += 1
                if not qa["is_impossible"]:
                    for answer in qa['answers']:
                        ans.append([answer['text'], answer["answer_start"], context, question])
    return ctx,que,ans


def coqa(dev=False):
    ctx,que,ans = list(), list(), list()
    context, question = -1, -1
    if not os.path.exists(COQA_TRAIN):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=TRAIN_SET['coqa'].split('/')[-1]) as t:    
            urllib.request.urlretrieve(TRAIN_SET['coqa'], COQA_TRAIN)
    if not os.path.exists(COQA_DEV):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=DEV_SET['coqa'].split('/')[-1]) as t:    
            urllib.request.urlretrieve(DEV_SET['coqa'], COQA_DEV)
    if dev:
        data = json.load(open(COQA_DEV))
    else:
        data = json.load(open(COQA_TRAIN))

    for entry in tqdm(data['data'], desc='PreProcessing Coqa'):
        ctx.append(entry['story'])
        context += 1
        for q,a in zip(entry['questions'], entry['answers']):
            que.append(q['input_text'])
            question += 1
            if(a['input_text'] == ''):
                ans.append(ctx[context][a['span_start']:a['span_end'], a['span_start'], context, question])
            else:
                ans.append([a['input_text'], a['span_start'], context, question])
    return ctx,que,ans 
        
def quac(dev=False):
    ctx,que,ans = list(), list(), list()
    context, question = -1, -1    
    if not os.path.exists(QUAC_TRAIN):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=TRAIN_SET['quac'].split('/')[-1]) as t:
            urllib.request.urlretrieve(TRAIN_SET['quac'], QUAC_TRAIN)
    if not os.path.exists(QUAC_DEV):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=DEV_SET['quac'].split('/')[-1]) as t:
            urllib.request.urlretrieve(DEV_SET['quac'], QUAC_DEV)

    if dev:
        data = json.load(open(QUAC_DEV))
    else:
        data = json.load(open(QUAC_TRAIN))

    for entry in tqdm(data['data'], desc='PreProcessing Quac'):
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
    return ctx, que, ans

def natural_questions(dev=False):
    ctx,que,ans = list(), list(), list()
    context, question = -1, -1
    print(tfds.list_builders())
    nq_builder = tfds.text.natural_questions.NaturalQuestions(data_dir='./data')
    info = nq_builder.info
    print(info)
        #if(dev):
    #    data = tfds.text.natural_questions.NaturalQuestions(data_dir='./data')
    #else:
    #    data = tfds.text.natural_questions.NaturalQuestions(data_dir='./data')
    
    #print(data)
    return ctx, que, ans

def preprocess(dev=False, *args):
    ctx, que, ans = list(), list(), list()
    for arg in args:
        if arg == 'squad':
            c,q,a = squad(dev)
        elif arg == 'coqa':
            c,q,a = coqa(dev)
        elif arg == 'quac':
            c,q,a = quac(dev)
        elif arg == 'natural_questions':
            c,q,a = natural_questions(dev)

        ctx.extend(c)
        que.extend(q)
        ans.extend(a)
    return ctx, que, ans