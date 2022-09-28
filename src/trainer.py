"""
trainer
"""

import os
import time
import torch
from transformers import BertPreTrainedModel, BertTokenizer, BertModel
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DataSet
from model import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

local_rank = int(os.environ["LOCAL_RANK"])

is_distributed = True

torch.cuda.set_device(local_rank)
if is_distributed:
    torch.distributed.init_process_group(backend='nccl')

device = torch.device("cuda:" + str(local_rank))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_name = "BiEncoderWithInBatchNegative"
model = BiEncoderWithInBatchNegative.from_pretrained("bert-base-uncased").to(device)
model.train()

if is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, eps=1e-8)
optimizer.zero_grad()

data_set = DataSet(tokenizer, device, rank=local_rank, limit=None)
train_dataloader = DataLoader(data_set, batch_size=None, num_workers=4)

scaler = torch.cuda.amp.GradScaler()

interval = 100
model_save_interval = 10000
steps = 0
total_loss = 0
start_time = time.time()
train_time = 0

optimizer.zero_grad()

#for batch in data_set:
for batch in train_dataloader:
    q_inputs, pos_inputs, neg_inputs = batch
    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
    pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
    neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}

    cur_time = time.time()

    with torch.cuda.amp.autocast():
        loss = model(q_inputs, pos_inputs, neg_inputs)

    total_loss += loss.item()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    steps += 1
    train_time += time.time() - cur_time
    if steps % interval == 0:
        print(steps, total_loss / interval, time.time() - start_time, train_time)
        #torch.cuda.empty_cache()
        start_time = time.time()
        train_time = 0
        total_loss = 0

    if steps % model_save_interval == 0:
        if is_distributed:
            model.module.save_pretrained("bert-base-" + model_name)
        else:
            model.save_pretrained("bert-base-" + model_name)

if is_distributed:
    model.module.save_pretrained("bert-base-" + model_name)
else:
    model.save_pretrained("bert-base-" + model_name)
