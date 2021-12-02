#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-19 9:08 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import numpy as np


def save_data():
    """@Numpy Examples
    数据的保存与加载

    Examples:
        # 保存 array
        >>> a = np.asarray([1, 2, 3])
        >>> fp = r'./array.npy'  # 后缀为 .npy
        >>> np.save(fp, a)
        >>> _a = np.load(fp)
        >>> assert (a == _a).all()
        >>> _ = os.system(f'rm {fp}')

        # 保存多个数据，不限于 array
        >>> a = np.asarray([1, 2, 3])
        >>> b = '测试'
        >>> fp = r'./data.npz'  # 后缀为 .npz
        >>> np.savez(fp, a=a, b=b)
        >>> d = np.load(fp)
        >>> assert (a == d['a']).all() and b == d['b']
        >>> _ = os.system(f'rm {fp}')
    """


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
