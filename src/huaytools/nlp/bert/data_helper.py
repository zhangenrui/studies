#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-21 11:33 上午

Author: huayang

Subject:
    快速构建训练数据
"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict
from dataclasses import dataclass

# from sentence_transformers import InputExample


@dataclass(init=True)
class InputItem:
    """"""
    pid: str = None
    texts: List[str] = None
    label: Union[int, float] = None

    def __post_init__(self):
        """"""


def _test():
    """"""
    doctest.testmod()

    it = InputItem()


if __name__ == '__main__':
    """"""
    _test()
