import os
import re
import json
import zipfile
import subprocess
import numpy as np
from tqdm.auto import tqdm
import urllib.request
import tensorflow_datasets as tfds
import spacy
import collections
import glob
from gzip import GzipFile
import json
import multiprocessing

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

NQ_SP_TRAIN_URL = 'https://storage.cloud.google.com/natural_questions/v1.0/sample/nq-train-sample.jsonl.gz'
NQ_SP_DEV_URL = 'https://storage.cloud.google.com/natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz'
NQ_SP_TRAIN = "./data/v1.0_sample_nq-train-sample.jsonl.gz"
NQ_SP_DEV = "./data/v1.0_sample_nq-dev-sample.jsonl.gz"

nlp = spacy.load('en_core_web_lg')

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
            model = Fasttext.FastTextKeyedVectors.load_word2vec_format('./data/wiki.en.vec', binary=False, encoding='utf8')
            word_vectors = model.wv
            del model
            return word_vectors
 
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


def natural_questions(dev=False, Large=False, long=False):
    ctx,que,ans_l,ans_s = list(), list(), list(), list()
    context, question = -1, -1
    if not os.path.exists(NQ_SP_DEV):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=NQ_SP_DEV_URL.split('/')[-1]) as t:    
                urllib.request.urlretrieve(NQ_SP_DEV_URL, NQ_SP_DEV)  
    if not os.path.exists(NQ_SP_TRAIN):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=NQ_SP_TRAIN_URL.split('/')[-1]) as t:    
                urllib.request.urlretrieve(NQ_SP_TRAIN_URL, NQ_SP_TRAIN)  
    if Large:
        p = subprocess.Popen("gsutil -m cp -R gs://natural_questions/v1.0 ./data", shell=True)
     
    html_re = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
   
    if(dev):
        file = open(NQ_SP_DEV, 'rb')
    else:
        file = open(NQ_SP_TRAIN, 'rb')

    import enum
    
    class AnswerType(enum.IntEnum):
        UNKNOWN = 0
        YES = 1
        NO = 2
        SHORT = 3
        LONG = 4

    class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
        def __new__(cls, type_, text=None, offset=None):
            return super(Answer, cls).__new__(cls, type_, text, offset)

    class NqExample(object):        
        def __init__(self, example_id, qas_id, questions, doc_tokens, doc_tokens_map=None, answer=None, start_position=None, end_position=None):
            self.example_id = example_id
            self.qas_id = qas_id
            self.questions = questions
            self.doc_tokens = doc_tokens
            self.doc_tokens_map = doc_tokens_map
            self.answer = answer
            self.start_position = start_position
            self.end_position = end_position

    def should_skip_context(e, idx):
        if (not e["long_answer_candidates"][idx]["top_level"]):
            return True
        elif not get_candidate_text(e, idx).text.strip():
            return True
        else:
            return False

    def get_first_annotation(e):
        positive_annotations = sorted( [a for a in e["annotations"] if has_long_answer(a)], key=lambda a: a["long_answer"]["candidate_index"])
        for a in positive_annotations:
            if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"]
            return a, idx, (token_to_char_offset(e, idx, start_token), token_to_char_offset(e, idx, end_token) - 1)

        for a in positive_annotations:
            idx = a["long_answer"]["candidate_index"]
            return a, idx, (-1, -1)

        return None, -1, (-1, -1)

    def get_text_span(example, span):
        token_positions = []
        tokens = []
        for i in range(span["start_token"], span["end_token"]):
            t = example["document_tokens"][i]
            if not t["html_token"]:
            token_positions.append(i)
            token = t["token"].replace(" ", "")
            tokens.append(token)
        return TextSpan(token_positions, " ".join(tokens))

    def token_to_char_offset(e, candidate_idx, token_idx):
        c = e["long_answer_candidates"][candidate_idx]
        char_offset = 0
        for i in range(c["start_token"], token_idx):
            t = e["document_tokens"][i]
            if not t["html_token"]:
            token = t["token"].replace(" ", "")
            char_offset += len(token) + 1
        return char_offset

    def get_candidate_type(e, idx):
        if first_token == "<Table>":
            return "Table"
        elif first_token == "<P>":
            return "Paragraph"
        elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
            return "List"
        elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
            return "Other"
        else:
            print("Unknoww candidate type found: %s", first_token)
            return "Other"

    def add_candidate_types_and_positions(e):
        for idx, c in candidates_iter(e):
            context_type = get_candidate_type(e, idx)
            if counts[context_type] < FLAGS.max_position:
                counts[context_type] += 1
            c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])
    
    def get_candidate_type_and_position(e, idx):
        if idx < 0 or idx >= len(e["long_answer_candidates"]):
            return TextSpan([], "")
        return get_text_span(e, e["long_answer_candidates"][idx])

    def candidates_iter(e):
        for idx, c in enumerate(e["long_answer_candidates"]):
            if should_skip_context(e, idx):
                continue
        yield idx, c

    def create_example_from_jsonl(line):
        e = json.loads(line, object_pairs_hook=collections.OrderedDict)
        add_candidate_types_and_positions(e)
        annotation, annotated_idx, annotated_sa = get_first_annotation(e)
        question = {"input_text": e["question_text"]}
        answer = { "candidate_id": annotated_idx, "span_text": "", "span_start": -1, "span_end": -1, "input_text": "long"}
        if annotation is not None:
            assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
            if annotation["yes_no_answer"] in ("YES", "NO"):
                answer["input_text"] = annotation["yes_no_answer"].lower()
        if annotated_sa != (-1, -1):
            answer["input_text"] = "short"
            span_text = get_candidate_text(e, annotated_idx).text
            answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
            answer["span_start"] = annotated_sa[0]
            answer["span_end"] = annotated_sa[1]
            expected_answer_text = get_text_span(e, {
                    "start_token": annotation["short_answers"][0]["start_token"],
                    "end_token": annotation["short_answers"][-1]["end_token"],
                }).text
            assert expected_answer_text == answer["span_text"], (expected_answer_text, answer["span_text"])
        elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
            answer["span_text"] = get_candidate_text(e, annotated_idx).text
            answer["span_start"] = 0
            answer["span_end"] = len(answer["span_text"])
        context_idxs = [-1]
        context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
        context_list[-1]["text_map"], context_list[-1]["text"] = (get_candidate_text(e, -1))
        for idx, _ in candidates_iter(e):
            context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
            context["text_map"], context["text"] = get_candidate_text(e, idx)
            context_idxs.append(idx)
            context_list.append(context)
            if len(context_list) >= FLAGS.max_contexts:
                break
        example = {
            "name": e["document_title"],
            "id": str(e["example_id"]),
            "questions": [question],
            "answers": [answer],
            "has_correct_context": annotated_idx in context_idxs
        }

        single_map = []
        single_context = []
        offset = 0
        for context in context_list:
            single_map.extend([-1, -1])
            single_context.append("[ContextId=%d] %s" % (context["id"], context["type"]))
            offset += len(single_context[-1]) + 1
            if context["id"] == annotated_idx:
                answer["span_start"] += offset
                answer["span_end"] += offset

            if context["text"]:
                single_map.extend(context["text_map"])
                single_context.append(context["text"])
                offset += len(single_context[-1]) + 1

        example["contexts"] = " ".join(single_context)
        example["contexts_map"] = single_map
        if annotated_idx in context_idxs:
            expected = example["contexts"][answer["span_start"]:answer["span_end"]]
            assert expected == answer["span_text"], (expected, answer["span_text"])

        return example

    def make_nq_answer(contexts, answer):
        start = answer["span_start"]
        end = answer["span_end"]
        input_text = answer["input_text"]

        if (answer["candidate_id"] == -1 or start >= len(contexts) or
            end > len(contexts)):
            answer_type = AnswerType.UNKNOWN
            start = 0
            end = 1
        elif input_text.lower() == "yes":
            answer_type = AnswerType.YES
        elif input_text.lower() == "no":
            answer_type = AnswerType.NO
        elif input_text.lower() == "long":
            answer_type = AnswerType.LONG
        else:
            answer_type = AnswerType.SHORT
        return Answer(answer_type, text=contexts[start:end], offset=start)
    
    def read_nq_entry(entry):
        def is_whitespace(c):
            return c in " \t\r\n" or ord(c) == 0x202F
        examples = []
        contexts_id = entry["id"]
        contexts = entry["contexts"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in contexts:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        questions = []
        for i, question in enumerate(entry["questions"]):
            qas_id = "{}".format(contexts_id)
            question_text = question["input_text"]
            start_position = None
            end_position = None
            answer = None
            if dev:
                answer_dict = entry["answers"][i]
                answer = make_nq_answer(contexts, answer_dict)
                if answer is None or answer.offset is None:
                    continue
                start_position = char_to_word_offset[answer.offset]
                end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(answer.text))
            if actual_text.find(cleaned_answer_text) == -1:
                print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue

            questions.append(question_text)
            example = NqExample(example_id=int(contexts_id), qas_id=qas_id, questions=questions[:],doc_tokens=doc_tokens,
                doc_tokens_map=entry.get("contexts_map", None), answer=answer, start_position=start_position, end_position=end_position)
            examples.append(example)
        return examples


    annotation_dict = {}       

    input_data = [] 
    with GzipFile(fileobj=file) as input_file:
        for line in input_file: #tqdm(input_file, desc='PreProcessing NQ'):
            input_data.append(create_example_from_jsonl(line))
    examples = []
    for entry in input_data:
        examples.extend(read_nq_entry(entry))
    for example in examples:
        ctx.append(' '.join([t for t in example.doc_tokens]))
        context += 1
        for q in example.questions:
            que.append(q)
            question += 1
        ans_s.append([example.answer, example.start_position, ])
        

            '''json_example = json.loads(line)
            example_id = json_example['example_id']
            document_tokens = json_example['document_tokens']
            ctx.append(" ".join([re.sub(" ", "_", t['token']) for t in json_example['document_tokens'] if t['html_token'] == False]))
            context += 1
            que.append(json_example['question_text'])
            question += 1
            canidates = []
            for annotation in json_example['annotations']:
                if(len(annotation['long_answer']) > 0 and annotation['long_answer']['canidate_index'] not in canidates):
                    long_token = annotation['long_answer']
                    start_token = long_token['start_token']
                    end_token = long_token['end_token']
                    canidates.append(long_token['candidate_index'])
                    long_answer = ' '.join([re.sub(" ", "_", t['token']) for t in document_tokens[start_token: end_token] if t['html_token'] == False]) 
                    start = 0
                    for t in document_tokens[:start_token]:
                        if(t['html_token'] == False):
                            start += 1
                    if(long_answer != ''):
                        ans_l.append([long_answer, start, context, question])
                for short_span_rec in annotation['short_answers']:
                    start_token = short_span_rec['start_token']
                    end_token = short_span_rec['end_token']
                    short_answer = ' '.join([re.sub(" ", "_", t['token']) for t in document_tokens[start_token: end_token] if t['html_token'] == False])
                    start = 0
                    for t in document_tokens[:start_token]:
                        if(t['html_token'] == False):
                            start += 1
                    if(short_answer != ''):
                        ans_s.append([short_answer, start, context, question])
            for i, c in enumerate(json_example['long_answer_candidates']):
                if(i not in canidates):
                    start_token = long_token['start_token']
                    end_token   = long_token['end_token']
                    long_answer = ' '.join([re.sub(" ", "_", t['token']) for t in document_tokens[start_token: end_token] if t['html_token'] == False]) 
                    start = 0
                    for t in document_tokens[:start_token]:
                        if(t['html_token'] == False):
                            start += 1
                    if(long_answer != ''):
                        ans_l.append([long_answer, start, context, question])
    if(long):
        ans_s.extend(ans_l)     '''           

    return ctx, que, ans_s

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

if __name__ == "__main__":
    preprocess(True, 'natural_questions')