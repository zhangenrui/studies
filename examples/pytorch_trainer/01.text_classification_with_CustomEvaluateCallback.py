#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-20 7:17 下午

Author: huayang

Subject:

"""
import doctest

from typing import *

import torch
import torch.nn as nn

from pytorch_trainer import Trainer, EvaluateCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BERT_MODEL_NAME = r'bert-base-chinese'


class BertClassification(nn.Module):
    """模型"""

    def __init__(self):
        """"""
        super().__init__()
        from transformers.models.bert import BertModel

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.clf = nn.Linear(768, 2)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, labels=None):
        """"""
        x = self.bert(inputs)[1]  # cls
        x = self.dropout(x)
        logits = self.clf(x)
        probs = self.softmax(logits)

        if labels is not None:
            loss = self.cross_entropy(logits, labels)
            return probs, loss

        return probs


class MyEvaluateCallback(EvaluateCallback):
    """评估器"""

    def compute_metrics(self, model, val_data_loader) -> Dict[str, float]:
        """"""
        val_loss = 0
        batch_cnt = 0
        y_true = []
        y_pred = []
        for batch in val_data_loader:
            inputs, labels = batch
            batch_cnt += 1
            prob, loss = model(inputs, labels)
            val_loss += loss.mean().numpy().item()

            y_true.append(labels)
            y_pred.append(torch.argmax(prob, -1))

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        val_loss /= batch_cnt
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)

        return {
            'loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class MyTrainer(Trainer):
    """"""

    def training_step(self, model, batch):
        """"""
        inputs, labels = batch
        probs, loss = model(inputs, labels)
        return probs, loss

    def set_data_loader(self, batch_size, device):
        """"""
        from itertools import islice
        from huaytools.nlp.utils import split
        from huaytools.pytorch.data._basic import ToyDataLoader
        from transformers.models.bert.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

        fp = r'./data_files/sentiment_demo_100.txt'
        # fp = r'/Users/huayang/Downloads/for_polarity_fine_tune_hotel_1018.txt'
        max_len = 32
        inputs, labels = [], []
        with open(fp, encoding='utf8') as f:
            for ln in islice(f, None):
                txt, label = ln.strip().split('\t')
                _, token_ids = tokenizer.encode(txt, padding='max_length', truncation=True, max_length=max_len)
                inputs.append(token_ids)
                labels.append(int(label))

        ds_train, ds_val = split(inputs, labels, split_size=0.3)
        self.train_data_loader = ToyDataLoader(ds_train, batch_size=batch_size, device=device)
        self.val_data_loader = ToyDataLoader(ds_val, batch_size=batch_size, device=device, shuffle=False)

    def set_model(self):
        """"""
        self.model = BertClassification()


def _test():
    """"""
    doctest.testmod()


def main():
    """"""
    trainer = MyTrainer(use_cpu_device=True, evaluate_callback=MyEvaluateCallback())
    trainer.num_gradient_accumulation = 3
    trainer.batch_size = 8  # batch_size = 24
    trainer.num_evaluate_per_steps = 5
    trainer.num_train_epochs = 10

    trainer.train()


if __name__ == '__main__':
    """"""
    # _test()
    main()
