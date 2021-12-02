#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-08 2:56 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

from huaytools.pytorch.train import Trainer
from huaytools.pytorch.nn import build_bert_pretrained, BertConfig
from huaytools.pytorch.nn import tokenizer

from chinese_sentence_segmentation.model import SegmentModel1


class MyTrainer(Trainer):
    """"""

    def set_model(self):
        ckpt_path = r'../../ckpt/bert-base-chinese'
        args = BertConfig()
        bert = build_bert_pretrained(ckpt_path)
        self.model = SegmentModel1(bert, args)

    def set_data_loader(self, batch_size, device):
        pass


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
