
import squad_dataloader as squad

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader



def train(dataset):
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for x, y in dataset:
        x = x
        
def main():
    data = squad.Squad_DataSet(train = False)
    train(data)
    
    val_data = squad.Squad_DataSet(train=False)
    
   



if __name__ == "__main__":
    main()