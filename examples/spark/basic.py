#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-19 3:15 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import pyspark


def list_dir(sc, dir_path='.'):
    """展示 executor 机器上的目录结构"""
    def _list_dir(idx):  # noqa
        """"""
        import os

        fns = os.listdir(dir_path)

        return fns

    rdd = sc.parallelize([1])
    rdd.map(_list_dir).collect()


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
