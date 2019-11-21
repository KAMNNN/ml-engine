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

from squad_utils import (read_squad_examples, convert_examples_to_features,RawResult, write_predictions, RawResultExtended, write_predictions_extended)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ner_whitelist = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']


def PreProcess(datadir='./data', train=True):
    TrainSet = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    TestSet = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    urllib.request.urlretrieve(TrainSet, os.path.join(datadir,'squad-train-v2.0.json')) 
    urllib.request.urlretrieve(TestSet, os.path.join(datadir,'squad-dev-v2.0.json'))

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if(train):
        input_file = os.path.join(datadir, 'squad-train-v2.0.json')
    else:            
        input_file = os.path.join(datadir, 'squad-dev-v2.0.json')
            
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')        
    examples = read_squad_examples(input_file=input_file, is_training=train, version_2_with_negative=True)
    features = convert_examples_to_features(examples, tokenizer, 384, 128, 64, train)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if train:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                        all_start_positions, all_end_positions,\
                        all_cls_index, all_p_mask)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids,\
                        all_example_index, all_cls_index, all_p_mask)
        