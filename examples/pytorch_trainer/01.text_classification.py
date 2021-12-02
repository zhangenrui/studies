#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-20 7:17 下午

Author: huayang

Subject: Pytorch Trainer Demo

"""
import os
import sys
import json
import logging
import doctest

from typing import *
from collections import defaultdict

import torch
import torch.nn as nn

DEFAULT_LOG_FMT = '%(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=DEFAULT_LOG_FMT, datefmt='%Y.%m.%d %H:%M:%S', level=logging.INFO)

from pytorch_trainer import Trainer, EvaluateCallback  # noqa

BERT_MODEL_NAME = r'bert-base-chinese'


class BertClassification(nn.Module):
    """Bert 文本分类，使用 transformers.BertModel"""

    def __init__(self, n_classes=2, dropout=0.1):
        """"""
        super().__init__()
        from transformers.models.bert import BertModel

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(768, n_classes)
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


class MyTrainer(Trainer):
    """"""

    # 优化了默认 training_step 的逻辑，根据 batch 的类型推断 model 编码 batch 的方式，
    # 如果默认无法满足，需要重写 training_step，如下：
    # def training_step(self, model, batch):
    #     """"""
    #     inputs, labels = batch
    #     probs, loss = model(inputs, labels)
    #     return probs, loss

    def set_model(self):
        """"""
        self.model = BertClassification()

    def set_data_loader(self, batch_size):
        """"""
        from itertools import islice
        from huaytools.nlp.utils import split
        from huaytools.pytorch.data import ToyDataLoader
        from transformers.models.bert.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

        fp = r'./data_files/sentiment_demo_100.txt'
        # fp = r'/Users/huayang/Downloads/for_polarity_fine_tune_hotel_1018.txt'
        max_len = 32
        inputs, labels = [], []
        with open(fp, encoding='utf8') as f:
            for ln in islice(f, None):
                txt, label = ln.strip().split('\t')
                token_ids = tokenizer.encode(txt, padding='max_length', truncation=True, max_length=max_len)
                inputs.append(token_ids)
                labels.append(int(label))

        ds_train, ds_val = split(inputs, labels, split_size=0.2)
        self.train_data_loader = ToyDataLoader(ds_train, batch_size=batch_size)
        self.val_data_loader = ToyDataLoader(ds_val, batch_size=batch_size, shuffle=False)


def _test():
    """"""
    doctest.testmod()


def main():
    """"""
    # 默认 early_stopping=False；
    evaluate_callback = EvaluateCallback(early_stopping=True,
                                         evaluate_before_train=True,
                                         baseline=0.2)

    trainer = MyTrainer(batch_size=8,
                        num_gradient_accumulation=2,
                        num_train_epochs=20,
                        use_cpu_device=True,
                        evaluate_callback=evaluate_callback,
                        optimizer_type=torch.optim.AdamW,
                        random_seed=123,
                        save_dir=r'/Users/huayang/out/models/demo-text_clf')
    trainer.train()

    del trainer

    # continue training
    trainer = MyTrainer.from_trained(r'/Users/huayang/out/models/demo-text_clf')
    trainer.num_train_epochs = 3
    trainer.num_warmup_steps = 0
    trainer.train()


if __name__ == '__main__':
    """"""
    # _test()
    main()

# seed = 123
"""
INFO - MyTrainer - Config(15): {
    "batch_size": 8,
    "device": "cpu",
    "learning_rate": 5e-05,
    "no_decay_params": [
        "bias",
        "LayerNorm.weight"
    ],
    "num_gradient_accumulation": 2,
    "num_train_epochs": 20,
    "num_train_steps": 100,
    "num_warmup_steps": 10.0,
    "optimizer_type": "torch.optim.adamw.AdamW",
    "random_seed": 123,
    "save_dir": "/Users/huayang/out/models/demo-text_clf",
    "save_model_old_format": false,
    "save_model_state_dict": true,
    "use_cpu_device": true,
    "weight_decay": 0.01
}
INFO - MyTrainer - Train data size: 80
INFO - MyTrainer - Val data size: 20
INFO - EvaluateCallback - Evaluate metric: val_loss=0.71 before training
INFO - EvaluateCallback - Evaluate metric: val_loss=0.748 after epoch(0)
INFO - MyTrainer - model saved at /Users/huayang/out/models/demo-text_clf/model.pt
INFO - EvaluateCallback - Evaluate metric: val_loss=0.627 after epoch(1)
INFO - MyTrainer - model saved at /Users/huayang/out/models/demo-text_clf/model.pt
INFO - EvaluateCallback - Evaluate metric: val_loss=0.428 after epoch(2)
INFO - MyTrainer - model saved at /Users/huayang/out/models/demo-text_clf/model.pt
INFO - EvaluateCallback - Evaluate metric: val_loss=0.369 after epoch(3)
INFO - MyTrainer - model saved at /Users/huayang/out/models/demo-text_clf/model.pt
INFO - EvaluateCallback - Evaluate metric: val_loss=0.262 after epoch(4)
INFO - MyTrainer - model saved at /Users/huayang/out/models/demo-text_clf/model.pt
INFO - EvaluateCallback - Evaluate metric: val_loss=0.0586 after epoch(5)
INFO - MyTrainer - model saved at /Users/huayang/out/models/demo-text_clf/model.pt
INFO - EvaluateCallback - Evaluate metric: val_loss=0.331 after epoch(6)
INFO - EvaluateCallback - Evaluate metric: val_loss=0.295 after epoch(7)
INFO - EvaluateCallback - Early stop with best val_loss=0.0586 after global_step[40].
"""
