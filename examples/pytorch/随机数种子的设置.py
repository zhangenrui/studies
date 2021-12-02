#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-26 2:21 下午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa

# from typing import *
# from itertools import islice
# from collections import defaultdict

# from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from huaytools.pytorch.utils import set_seed


def _test_data_loader():
    """"""
    set_seed(123)
    dl1 = DataLoader(torch.arange(31), batch_size=8, shuffle=True, num_workers=0)
    for b in dl1:
        print(b)
    '''
    tensor([27,  3,  4, 20,  0,  7, 17,  8])
    tensor([ 6, 26, 23,  1, 21, 28, 16, 15])
    tensor([ 9,  2, 24, 13, 14, 10, 25,  5])
    tensor([18, 22, 12, 19, 11, 30, 29])
    '''
    set_seed(123)
    dl2 = DataLoader(torch.arange(31), batch_size=8, shuffle=True, num_workers=3)
    for b in dl2:
        print(b)
    # set_seed(123)
    # ds = torch.arange(31)
    # dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    # for b in dl:
    #     print(b)


def _test():
    """"""
    doctest.testmod()

    _test_data_loader()


if __name__ == '__main__':
    """"""
    _test()
