#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-03 1:05 下午

Author: huayang

Subject: 模型 tracing 化的基本流程，如果遇到与 if-语句相关的 TracerWarning，
    可以参考下面链接中与 torch.jit.script 相关的内容

References:
    [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import torch

from huaytools.pytorch.nn.bert import build_bert_pretrained


@torch.no_grad()
def trace_pipeline():
    """"""
    ckpt = r'../../ckpt/bert-base-chinese'
    model, tokenizer = build_bert_pretrained(ckpt, return_tokenizer=True)
    example_inputs = tokenizer.batch_encode(['测试1', '测试2'],
                                            return_token_masks=True, return_token_type_ids=True)
    model.eval()  # 没有这步，可能会报误差警告 TracerWarning
    traced_model = torch.jit.trace(model, tuple(example_inputs))

    save_path = r'../../ckpt/bert-base-chinese/traced_bert.pt'
    traced_model.save(save_path)

    loaded_model = torch.jit.load(save_path)
    inputs = tokenizer.batch_encode(['我爱NLP', '我爱Python'], convert_fn=torch.as_tensor,
                                    return_token_type_ids=True, return_token_masks=True)
    o = loaded_model(*inputs)
    print(o[0][0, :5])

    o2 = model(*inputs)
    print(o2[0][0, :5])
    """
    tensor([0.9975, 1.0000, 0.9740, 0.8419, 0.9781])
    tensor([0.9975, 1.0000, 0.9740, 0.8419, 0.9781])
    """


def _test():
    """"""
    trace_pipeline()


if __name__ == '__main__':
    """"""
    _test()
