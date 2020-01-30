import torch 
import numpy as np
from torch.utils.data import DataLoader
from transformers import *
from tqdm import tqdm
from model import *
import subprocess
import spacy
import data

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

#subprocess.call("python -m spacy download en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm
EPOCHS = 4


device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cpu")

def update(engine, batch):
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    lm_loss = model(*batch)
    loss = lm_loss / 8
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if engine.state.iteration % 8 == 0:
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

def main():
    ds = data.Bert_GPT2_DataClass()
    dl = DataLoader(ds, num_workers=4, batch_sampler=4)
    trainer = Engine(update)
    model = GPT2_QGEN().to(device)
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, 6.25e-5), (EPOCHS * len(dl), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = { "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1)) }
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])
    tb_logger = TensorboardLogger(log_dir='./logs')
    checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  
    torch.save( tb_logger.writer.log_dir + '/model_training.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))    
    tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    trainer.run(dl, max_epochs=EPOCHS)
    tb_loger.close()
if __name__ == "__main__":
    main()
