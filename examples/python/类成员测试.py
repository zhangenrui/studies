#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-03 2:48 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict


class A:
    L: list
    a: int = 3


class B:
    L: list = []


def _test():
    """"""
    # doctest.testmod()

    a1 = A()
    a1.a = 4
    a1.L = [1]

    a2 = A()
    print(a2.a)  # 3
    print(a2.L)  # None

    b1 = B()
    b1.L.append(1)

    b2 = B()
    print(b2.L)  # [1]


if __name__ == '__main__':
    """"""
    _test()
