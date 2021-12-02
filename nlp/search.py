#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-11 5:28 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict


__all__ = [
    'hard_match'
]


def hard_match(txt: str, tags: List[str], ordered=True):
    """
    如果 tags 中的每个 tag 都在 txt 中出现过，则返回 True

    Examples:
        >>> hard_match('12345', ['1', '3', '4'])
        True
        >>> hard_match('12345', ['1', '3', '7'])
        False
        >>> hard_match('12345', ['1', '5', '3'])
        False
        >>> hard_match('12345', ['1', '5', '3'], ordered=False)
        True
        >>> hard_match('12345', ['1', '5', '7'], ordered=False)
        False
    """
    idx = 0
    for tag in tags:
        if ordered:
            idx = txt.find(tag, idx)
        else:
            idx = txt.find(tag)

        if idx < 0:
            return False

    return True


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
