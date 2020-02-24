import os
import torch
import math
from pprint import pformat

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from pytorch_transformers import *
from tqdm import tqdm
import subprocess

import dataloader

import ignite
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from argparse import ArgumentParser
#subprocess.call("python -m spacy download en_core_web_lg", shell=True)
#python -m spacy download en_core_web_lg
#python -m spacy download en_core_web_sm

EPOCHS = 4
BATCH_SIZE = 1
ITERATION_STEP = 8
THREAD_NUM = 1
DISTRIBUTED = False


def train():
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.device_count() > 1 else "cpu")
    model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    DISTRIBUTED = args.local_rank != -1
    
    if DISTRIBUTED and torch.distributed.is_available():
        print("Distributed")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        #BATCH_SIZE *= 2

    def average_distributed_scalar(scalar):
        if(not DISTRIBUTED):
            return scalar
        scalar_t = torch.tensor(scalar, dtype=torch.float, device=device) / torch.distributed.get_world_size()
        torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
        return scalar_t.item()
        
    optimizer = AdamW(model.parameters(), lr=6.25e-5)

    ds = dataloader.Conv_GPT2_DataClass(tokenizer)
    v_ds= dataloader.Conv_GPT2_DataClass(tokenizer, dev=True)
    orig_added_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(dataloader.ATTR_SPECIAL_TOKENS) 
    if(num_added_tokens > 0):
        model.resize_token_embeddings(new_num_tokens = orig_added_tokens + num_added_tokens)
    model = model.to(device)

    train_sampler = torch.utils.data.distributed.DistributedSampler(ds) if DISTRIBUTED else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(v_ds) if DISTRIBUTED else None

    dl = DataLoader(ds, sampler=train_sampler, batch_size=BATCH_SIZE, shuffle= not DISTRIBUTED)  
    v_dl = DataLoader(v_ds, sampler=valid_sampler, shuffle=False)  
    
   
    def update(engine, batch):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        lm_loss, *_ = model(batch[0], token_type_ids=batch[1], lm_labels=batch[2])
        loss = lm_loss / ITERATION_STEP
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if engine.state.iteration % ITERATION_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            lm_logits, *_ = model(batch[0], token_type_ids=batch[1])
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = batch[2][..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    trainer = Engine(update)
    evaluator = Engine(inference)

    scheduler = PiecewiseLinear(optimizer, "lr", [(0, 6.25e-5), (EPOCHS * len(ds)//BATCH_SIZE, 0.0)])   
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(v_dl))
    #trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(v_dl))
    
    if DISTRIBUTED:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"]),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"])})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    if(args.local_rank in [0, -1]):
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))  

        tb_logger = TensorboardLogger(log_dir='./logs')
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
        
        checkpoint_handler = ModelCheckpoint('./checkpoint', '_checkpoint', n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'gpt2_qg': getattr(model, 'module', model)})  
        
        getattr(model, 'module', model).config.to_json_file(os.path.join('./checkpoint', 'config'))   
        tokenizer.save_pretrained('./checkpoint') 
    
    trainer.run(dl, max_epochs=EPOCHS)
    if(args.local_rank in [0, -1]):
        tb_logger.close()

if __name__ == '__main__':
    train()