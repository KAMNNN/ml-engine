import os
import torch 
from torch.utils.data import DataLoader
from transformers import *
from tqdm import tqdm
import subprocess
import dataloader

import ignite
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

#subprocess.call("python -m spacy download en_core_web_lg")
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm

EPOCHS = 4
BATCH_SIZE = 1
ITERATION_STEP = 8
THREAD_NUM = 4

def average_distributed_scalar(scalar):
    return scalar

def train():
    device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cpu")
    model = GPT2LMHeadModel .from_pretrained('gpt2').to(device)
    optimizer = AdamW(model.parameters(), lr=6.25e-5)

    ds = dataloader.Conv_GPT2_DataClass()
    v_dl = dataloader.Conv_GPT2_DataClass(dev=True)
    dl = DataLoader(ds, num_workers=min(THREAD_NUM, BATCH_SIZE), batch_size=BATCH_SIZE)  
    v_dl = DataLoader(v_ds, num_workers=min(THREAD_NUM, BATCH_SIZE))  
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, 6.25e-5), (EPOCHS * len(ds)//BATCH_SIZE, 0.0)])
    metrics = { "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1)) }
    metrics.update({ "average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"])  })
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])

    def update(engine, batch):
        model.train()        
        batch = tuple(t.to(device) for t in batch)
        lm_loss, logits, T = model(batch[0], token_type_ids=batch[1], labels=batch[2])
        loss = lm_loss / ITERATION_STEP
        loss.backward()
        if engine.state.iteration % ITERATION_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    
    def eval(engine, batch):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        lm_logits = model(batch[0], token_type_ids=batch[1])[0]
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return lm_logits_flat_shifted, lm_labels_flat_shifted

    trainer = Engine(update)
    evaluator = Engine(eval)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")    
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"]) 
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))   
    tb_logger = TensorboardLogger(log_dir='./logs')
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
    checkpoint_handler = ModelCheckpoint('./checkpoint', '_checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'gpt2_qg': getattr(model, 'module', model)})  
    getattr(model, 'module', model).config.to_json_file(os.path.join('./checkpoint', 'config'))    
    trainer.run(dl, max_epochs=EPOCHS)
    tb_logger.close()

if __name__ == '__main__':
    train()