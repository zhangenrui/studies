#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2022-03-09 5:16 下午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa

# from collections import defaultdict
# from itertools import islice
# from pathlib import Path
from typing import *

# from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = [
    'Mixup',
    'ManifoldMixup'
]


class Mixup(nn.Module):
    """@Pytorch Utils
    mixup 数据增强策略

    Examples:
        >>> x = torch.randn(3, 5)
        >>> y = F.one_hot(torch.arange(3)).to(torch.float32)
        >>> mixup = Mixup()
        >>> x, y_a, y_b = mixup(x, y)

    References:
        https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    """
    a: Optional[float] = None

    def __init__(self, alpha=1.0):
        super().__init__()
        assert alpha > 0.
        self.alpha = alpha

    def update_a(self):
        self.a = np.random.beta(self.alpha, self.alpha)

    def reset_a(self):
        self.a = None

    def forward(self, x, y):
        """"""
        self.update_a()
        batch_size = x.size(0)
        idx = np.random.permutation(batch_size)
        x = self.a * x + (1 - self.a) * x[idx]
        y_a, y_b = y, y[idx]
        return x, y_a, y_b

    def mixup_loss(self, loss_fn, x, y_a, y_b):
        """"""
        assert self.a is not None
        loss = self.a * loss_fn(x, y_a) + (1 - self.a) * loss_fn(x, y_b)
        self.reset_a()
        return loss


class ManifoldMixup(Mixup):
    """@Pytorch Utils
    manifold mixup 数据增强策略

    Examples:
        >>> x = torch.randn(3, 5)
        >>> y = F.one_hot(torch.arange(3)).to(torch.float32)
        >>> mixup = ManifoldMixup()
        >>> x_, y_ = mixup(x, y)

        ```python
        # How to use mixup in model.
        def forward(self, x, target=None, use_mixup=False, mixup_alpha=None):
            x = self.layer1(x)

            if use_mixup and self.training:
                x, target = mixup(x, target)

            x = self.layer2(x)

            if self.training:
                return x, target
            else:
                return x
        ```

    Notes:
        The Difference of Mixup and Manifold_Mixup?
        - Mixup use for input (before any hidden layer).
        - Manifold_Mixup use for hidden layer.

    References:
        https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py
        - mixup_process
    """

    def forward(self, x, y):
        """"""
        self.update_a()
        batch_size = x.size(0)
        idx = np.random.permutation(batch_size)
        x = self.a * x + (1 - self.a) * x[idx]
        y = self.a * y + (1 - self.a) * y[idx]
        return x, y


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
