"""
model
"""

import time
import torch
from transformers import BertPreTrainedModel, BertTokenizer, BertModel
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F

class BiEncoder(BertPreTrainedModel):
    """
    bi encoder
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.dim = 128
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.init_weights()

    def forward(self, q, d_pos, d_neg, is_test=False):
        """
        forward
        """
        if is_test:
            q_cls = self.bert(**q).last_hidden_state[:, 0, :]
            q_cls = self.linear(q_cls)
            return q_cls
        q_cls = self.bert(**q).last_hidden_state[:, 0, :]
        d_pos_cls = self.bert(**d_pos).last_hidden_state[:, 0, :]
        d_neg_cls = self.bert(**d_neg).last_hidden_state[:, 0, :]

        q_cls = self.linear(q_cls)
        d_pos_cls = self.linear(d_pos_cls)
        d_neg_cls = self.linear(d_neg_cls)

        pos_sim = F.cosine_similarity(q_cls, d_pos_cls)
        neg_sim = F.cosine_similarity(q_cls, d_neg_cls)
        loss = (neg_sim - pos_sim + 0.8).mean()
        return loss

class BiEncoderWithInBatchNegative(BertPreTrainedModel):
    """
    bi encoder with in batch negative
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.dim = 128
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.init_weights()

    def forward(self, q, d_pos, d_neg, is_test=False):
        """
        forward
        """
        if is_test:
            q_cls = self.bert(**q).last_hidden_state[:, 0, :]
            q_cls = self.linear(q_cls)
            q_cls = F.normalize(q_cls, p=2, dim=1)
            return q_cls
        q_cls = self.bert(**q).last_hidden_state[:, 0, :]
        d_pos_cls = self.bert(**d_pos).last_hidden_state[:, 0, :]
        d_neg_cls = self.bert(**d_neg).last_hidden_state[:, 0, :]

        q_cls = F.normalize(q_cls, p=2, dim=1)
        d_pos_cls = F.normalize(d_pos_cls, p=2, dim=1)
        d_neg_cls = F.normalize(d_neg_cls, p=2, dim=1)

        d_all = torch.cat([d_pos_cls, d_neg_cls], dim=0)
        d_all = torch.transpose(d_all, 0, 1)
        sim = torch.matmul(q_cls, d_all)

        labels = torch.arange(0, sim.shape[0], device=q_cls.device)
        loss = torch.nn.CrossEntropyLoss()(sim, labels)
        return loss
