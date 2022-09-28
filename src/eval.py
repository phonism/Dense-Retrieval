"""
index

"""
import os
import faiss
import numpy as np
import time
import torch
from transformers import BertPreTrainedModel, BertTokenizer, BertModel
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DataSet, EvalDataSet, CollectionDataSet
from model import BiEncoder

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

local_rank = 3
torch.cuda.set_device(local_rank)
device = torch.device("cuda:" + str(local_rank))
model_name = "BiEncoderWithInBatchNegative"


model = BiEncoder.from_pretrained("bert-base-" + model_name).to(device)
#model.save_pretrained("bert-base-bi-encoder")
model.eval()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

qid_emb = []
qid_dict = {}
qid_rev_dict = {}
def predict_query():
    """
    predict query
    """
    with torch.no_grad():
        data_set = EvalDataSet(tokenizer, device, rank=local_rank)
        dev_dataloader = DataLoader(data_set, batch_size=None, num_workers=0)

        cnt = 0

        for batch in dev_dataloader:
            q_inputs, qid_list = batch
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            pos_inputs = q_inputs
            neg_inputs = q_inputs
            q_cls = model(q_inputs, pos_inputs, neg_inputs, is_test=True)
            for idx, qid in enumerate(qid_list):
                # qid_emb[qid] = q_cls[idx].cpu().detach().numpy()
                qid_emb.append(q_cls[idx].cpu().detach().numpy())
                qid_dict[cnt] = qid
                qid_rev_dict[qid] = cnt
                cnt += 1

doc_emb = []
docid_dict = {}

def predict_doc():
    """
    predict doc
    """
    with torch.no_grad():
        doc_data_set = CollectionDataSet(tokenizer, device, rank=local_rank)
        dev_doc_dataloader = DataLoader(doc_data_set, batch_size=None, num_workers=4)

        step = 0

        start_time = time.time()

        for batch in dev_doc_dataloader:
            doc_inputs, docid_list = batch
            pos_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
            q_inputs = pos_inputs
            neg_inputs = pos_inputs
            doc_cls = model(q_inputs, pos_inputs, neg_inputs, is_test=True)
            for idx, docid in enumerate(docid_list):
                doc_emb.append(doc_cls[idx].cpu().detach().numpy())
                docid_dict[step] = docid
                step += 1
                if step % 10000 == 0:
                    print(step, time.time() - start_time)
                    start_time = time.time()
                #if step == 100000:
                    #return

predict_query()
predict_doc()

dim = 128

start_time = time.time()

qid_emb = np.array(qid_emb)
doc_emb = np.array(doc_emb)

index = faiss.IndexFlatL2(dim)

#measure = faiss.METRIC_L2
#param = "IVF100,PQ16"
#index = faiss.index_factory(dim, param, measure)
#index.train(doc_emb)

index.add(doc_emb)

print("build index done!")

dist, res = index.search(qid_emb, 10)

print(time.time() - start_time)

f = open("msmarco/dev.res", "w+")

for qid_idx in range(len(res)):
    qid = qid_dict[qid_idx]
    for idx, num in enumerate(res[qid_idx]):
        # print(qid, docid_dict[num], dist[qid][idx])
        # print(qid + "\t" + docid_dict[num] + "\t" + str(idx + 1))
        f.write(qid + "\t" + docid_dict[num] + "\t" + str(idx + 1) + "\n")

f.close()
