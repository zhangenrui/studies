#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-04 4:11 下午

Author: huayang

Subject: 观察 loss 的相关属性

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# 为了方便观察，尽量简化参数的数量，这里有两个参数
model = nn.Linear(2, 1, bias=False)
# model.weight = Parameter(torch.Tensor([[1, 2]]))  # 重新设置参数（法1）
model.weight.data = torch.Tensor([[1, 2]])  # 重新设置参数（法2）
optimizer = torch.optim.SGD(model.parameters(), 1)  # 定义优化器，方便观察，学习率设为 1

print(f'model.parameters:\n    {model.weight.data}')
print()
'''
model.parameters:
    tensor([[1., 2.]])
'''

# 计算损失
inputs = torch.ones(2, 2)
loss = (model(inputs) / 10).mean()

# 在 loss 反向传播前，grad 为 None
print(f'before backward grad:\n    {model.weight.grad}')  # None
print()
'''
before backward grad:
    None
'''

# 反向传播后，此时还没有更新梯度，即没有 optimizer.step()
loss.backward()
print(f'after backward grad:\n    {model.weight.grad}')
print()
'''
after backward grad:
    tensor([[0.1000, 0.1000]])
'''

# 再计算一次 loss，并反向传播，可以发现梯度是会累加的
# 在 pytorch 中，利用这个特性可以非常方便的实现梯度累加
loss = (model(inputs) / 10).mean()
loss.backward()
print(f'after backward twice grad:\n    {model.weight.grad}')
print()
'''
after backward twice grad:
    tensor([[0.2000, 0.2000]])
'''

# optimizer.step() 前后
print(f'before optimizer.step():\n    {model.weight.data}')
optimizer.step()
print(f'after optimizer.step():\n    {model.weight.data}')
print()
'''
before optimizer.step():
    tensor([[1., 2.]])
after optimizer.step():
    tensor([[0.8000, 1.8000]])
'''

# optimizer.zero_grad() 前后，可以发现不执行 zero_grad，即使更新梯度后人梯度依然存在
print(f'before optimizer.zero_grad():\n    {model.weight.grad}')
optimizer.zero_grad()  # 注意 zero_grad 后，梯度不再是 None 了，而是 0
print(f'after optimizer.zero_grad():\n    {model.weight.grad}')
'''
before optimizer.zero_grad():
    tensor([[0.2000, 0.2000]])
after optimizer.zero_grad():
    tensor([[0., 0.]])
'''
