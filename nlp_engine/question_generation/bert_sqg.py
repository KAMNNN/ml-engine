import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import *
from tqdm import tqdm
import spacy
import dataloader

import ignite
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

BERT_TYPE = "bert-base-uncased"
EPOCHS = 25
BATCH_SIZE = 4
ITERATION_STEP = 8

tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)

class Bert_SQG(nn.Module):
    def __init__(self):
        super(Bert_SQG, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_TYPE)
        self.bert.train()
        self.cls = nn.Linear(self.bert.config.hidden_size, tokenizer.vocab_size, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h = self.bert(x)
        h = self.softmax(self.cls(h[0]))
        return h
    
def train():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Bert_SQG()
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    ds = dataloader.BertSQG_DataClass()
    dl = DataLoader(ds, num_workers=4, batch_size=4)
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, 3e-5), (EPOCHS * len(ds)//BATCH_SIZE, 0.0)])
    metrics = { "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1)) }
 
    def update(engine, batch):
        model.train()        
        for i in range(0, len(batch)-1):
            x = batch[i].to(device)
            y = batch[i+1].to(device)
            y_prime = model(x)
            loss = criterion(y_prime[-1], y[-1]) / ITERATION_STEP
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if engine.state.iteration % ITERATION_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
            
    trainer = Engine(update)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")    
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])    
    tb_logger = TensorboardLogger(log_dir='./logs')
    checkpoint_handler = ModelCheckpoint('./checkpoint', '_checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'bert_sqg': getattr(model, 'module', model)})  
    trainer.run(dl, max_epochs=EPOCHS)
    tb_loger.close()

if __name__ == '__main__':
    train()
