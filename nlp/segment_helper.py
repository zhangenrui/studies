#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-11 5:21 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import jieba

# try:
#     jieba.enable_paddle()
#     USE_PADDLE = True
# except:
#     USE_PADDLE = False


def jieba_segment(txt, cut_all=False, HMM=True, use_paddle=False, return_list=True):  # noqa
    """"""
    if return_list:
        segment = jieba.lcut
    else:
        segment = jieba.cut
    return segment(txt, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
