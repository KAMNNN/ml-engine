import os
import torch 
from torch.utils.data import DataLoader
from transformers import *
from tqdm import tqdm
import subprocess
import spacy
import data
import ignite
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler


#subprocess.call("python -m spacy download en_core_web_lg")
nlp = spacy.load("en")
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm
EPOCHS = 4
BATCH_SIZE = 4

device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cpu")
model = GPT2Model.from_pretrained('gpt2').to(device)
optimizer = AdamW(model.parameters(), lr=6.25e-5)

def train():
    ds = data.Bert_GPT2_DataClass()
    dl = DataLoader(ds, num_workers=12, batch_size=BATCH_SIZE)    


    scheduler = PiecewiseLinear(optimizer, "lr", [(0, 6.25e-5), (EPOCHS * len(ds)//BATCH_SIZE, 0.0)])
    metrics = { "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1)) }
    
    def update(engine, batch):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        model.train()
        lm_loss = model(*batch)
        loss = lm_loss / 8
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if engine.state.iteration % 8 == 0:
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
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  
    getattr(model, 'module', model).config.to_json_file(os.path.join('./checkpoint', 'config'))    
    trainer.run(dl, max_epochs=EPOCHS)
    tb_loger.close()

if __name__ == '__main__':
    train()