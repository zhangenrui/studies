#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-12-23 5:58 下午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa

from typing import Sequence, List


# from itertools import islice
# from collections import defaultdict

# from tqdm import tqdm


def remove_duplicates(src: Sequence, ordered=True) -> List:
    """
    remove duplicates

    Args:
        src:
        ordered:

    Examples:
        >>> ls = [1,2,3,3,2,4,2,3,5]
        >>> remove_duplicates(ls)
        [1, 2, 3, 4, 5]

    """
    ret = list(set(src))

    if ordered:
        ret.sort(key=src.index)

    return ret


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
