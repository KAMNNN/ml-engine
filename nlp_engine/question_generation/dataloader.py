import torch
import torch.nn as nn
import spacy
from tqdm import tqdm
from collections import defaultdict
from transformers import *
from torch.utils.data import Dataset   
from multiprocessing import Pool
import multiprocessing as mp
import datasets

vectorize = datasets.vectorize
THREAD_NUM = mp.cpu_count() // 2
nlp = spacy.load("en_core_web_lg")

class DataClass(Dataset): 
    def __init__ (self):
        self.contexts, self.questions, self.answers = datasets.preprocess(False, 'squad')
    def __len__(self):
        return len(self.answers)
    def __getitem__(self, idx):
        pass

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
        X = [input_tokens + mask_tokens]
        for i in range(len(output_tokens)):
            tokens = div_tokens[-1]
            tokens[-1] = output_tokens[i]
            X.append(tokens+mask_tokens)
        
        x = list()
        for _x in X:
            tensor = torch.zeros(self.max_size, dtype=torch.long)
            for i in range(len(_x)):
                tensor[i] = _x[i]
            x.append(tensor)
        y = torch.tensor(output_tokens, dtype=torch.long)        
        return x, y 
                    
class Conv_GPT2_DataClass(DataClass):
    def __init__(self):
        super(Conv_GPT2_DataClass, self).__init__()
        #self.tokenizer_ = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer1 = GPT2Tokenizer.from_pretrained("gpt2")
        self.SPECIAL_TOKENS = [ "<bos>", "<eos>", "<paragraph>", "<answer-general>", "<answer-specific>", "<question-general>", "<question-specific>", "<pad>" ]
        self.MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
        self.truncated_sequences = 0

    def get_position(self, para_ids, ans_ids, ans_prefix_ids):
        diff_index = -1
        for i, (pid, apid) in enumerate(zip(para_ids, ans_prefix_ids)):
            if pid != apid:
                diff_index = i
                break
        if diff_index == -1:
            diff_index = min(len(ans_prefix_ids), len(para_ids))
        return (diff_index, min(diff_index + len(ans_ids), len(para_ids)))

    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        bos, eos, paragraph, answer_general, answer_specific, question_general, question_specific = self.tokenizer1.convert_tokens_to_ids(self.SPECIAL_TOKENS[:-1])
        answer, start, context_id, question_id = self.answers[idx]
        
        context_tokens = self.tokenizer1.tokenize(self.contexts[context_id])
        question_tokens = self.tokenizer1.tokenize(self.questions[question_id])
        answer_tokens = self.tokenizer1.tokenize(answer)
        answer_token_prefix = self.tokenizer1.tokenize(self.contexts[context_id][:start])
        
        total_seq_len = len(context_tokens) + len(answer_tokens) + len(question_tokens) + 4
        if total_seq_len > self.tokenizer1.max_len: # Heuristic to chop off extra tokens in paragraphs            
            context_tokens = context_tokens[:-1 * (total_seq_len - self.tokenizer1.max_len + 1)]
            self.truncated_sequences += 1
            assert len(context_tokens) + len(answer_tokens) + len(question_tokens) + 4 < tokenizer.max_len

        context_tokens = self.tokenizer1.convert_tokens_to_ids(context_tokens)
        question_tokens = self.tokenizer1.convert_tokens_to_ids(question_tokens)
        answer_tokens = self.tokenizer1.convert_tokens_to_ids(answer_tokens)
        answer_token_prefix = self.tokenizer1.convert_tokens_to_ids(answer_token_prefix)
        
        token_start, token_end = self.get_position(context_tokens, answer_tokens, answer_token_prefix)


        sequence = [bos] + context_tokens
        token_types = [ answer_general if ((i - 1) >= token_start and (i - 1) < token_end) else paragraph  for i in range(len(context_tokens) + 1)]
        lm_labels = [-1 for _ in range(len(context_tokens)+1)]
        
        sequence.extend([answer_general] + answer_tokens)
        token_types.extend([answer_general for _ in range(len(answer_tokens) + 1)])
        lm_labels.extend([-1 for _ in range(len(answer_tokens) + 1)])

        sequence.extend([question_general] + question_tokens + [eos])
        token_types.extend([question_general for _ in range(len(question_tokens) + 2)])        
        lm_labels.extend([-1] + question_tokens + [eos])
        assert len(sequence) == len(token_types)
        assert len(token_types) == len(lm_labels)
        
        instance = {
            "input_ids": sequence,
            "token_type_ids": token_types,
            "lm_labels": lm_labels
        }

        padding = self.tokenizer1.convert_tokens_to_ids(self.SPECIAL_TOKENS[-1])
        max_l = self.tokenizer1.max_len
        out = list()
        for name in instance.keys():     
            out.append(torch.tensor( instance[name] + [padding if name != 'lm_labels' else -1] * (max_l - len(instance[name])), dtype=torch.long))
        
        return out