#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-09 10:26 上午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import torch
import torch.nn as nn


class TmpModel(nn.Module):
    """"""

    def __init__(self):
        """"""
        super().__init__()

        self.dense1 = nn.Linear(2, 1, bias=False)
        self.dense2 = nn.Linear(2, 1, bias=False)

        # dense1 和 dense2 共享参数
        share_weight = torch.Tensor([[1, 2]])
        self.dense1.weight.data = share_weight
        self.dense2.weight.data = share_weight

    def forward(self, inputs):  # [2, 2]
        x = self.dense1(inputs)  # [2, 1]
        x = x.expand(2, 2)  # [2, 2]
        x = self.dense2(x)  # [2, 1]

        loss = (x / 10).mean()
        return loss


model = TmpModel()
optimizer = torch.optim.SGD(model.parameters(), 1)  # 定义优化器，方便观察，学习率设为 1

print(f'model.parameters:\n    {list(model.parameters())}')
print()
'''
model.parameters:
    [Parameter containing:
tensor([[1., 2.]], requires_grad=True), Parameter containing:
tensor([[1., 2.]], requires_grad=True)]
'''

inputs = torch.ones(2, 2)
loss = model(inputs)

loss.backward()

print(f'after backward grad(dense1):\n    {model.dense1.weight.grad}')
print(f'after backward grad(dense2):\n    {model.dense2.weight.grad}')
print()
'''
after backward grad(dense1):
    tensor([[0.3000, 0.3000]])
after backward grad(dense2):
    tensor([[0.3000, 0.3000]])
'''

# 可以看到，对于共享参数，梯度是会累加的
#   dense1 和 dense2 的梯度都是 [0.3000, 0.3000]
#   但是最终更新的梯度是 [0.6000, 0.6000]
optimizer.step()
print(f'model.parameters:\n    {list(model.parameters())}')
'''
model.parameters:
    [Parameter containing:
tensor([[0.4000, 1.4000]], requires_grad=True), Parameter containing:
tensor([[0.4000, 1.4000]], requires_grad=True)]
'''
