import squad
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import *


def train(dataset):
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    gpt2 = GPT2Model.from_pretrained("gpt2-large")
    bert = BertModel.from_pretrained("bert-large-uncased")

    for ipt in dataset:
        pass
        
def main():
    data = squad.Squad_DataSet()
    train(data)
    
    val_data = squad.Squad_DataSet(train=False)
    
if __name__ == "__main__":
    main()