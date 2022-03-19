#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2022-03-19 4:29 下午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa


# from collections import defaultdict
# from itertools import islice
# from pathlib import Path
# from typing import *

# from tqdm import tqdm


class Script:
    """"""

    dp = r'/Users/huayang/workspace/my/studies/algorithms/problems'

    def process_one(self, fp):
        """"""
        f = open(fp, encoding='utf8').read()
        # print(f)
        fw = open()


def _test():
    """"""
    doctest.testmod()

    sc = Script()
    fp = r'/Users/huayang/workspace/my/studies/algorithms/problems/2022/03/LeetCode_0020_简单_有效的括号.md'
    sc.process_one(fp)



if __name__ == '__main__':
    """"""
    _test()
