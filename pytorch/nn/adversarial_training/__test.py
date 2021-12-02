#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-04 5:42 下午

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

from huaytools.pytorch.utils import set_seed
from huaytools.pytorch.nn import FGM, PGM

set_seed(1024)


def _test():
    """"""
    # doctest.testmod()

    model = nn.Linear(2, 1)
    print(list(model.parameters()))


if __name__ == '__main__':
    """"""
    _test()
