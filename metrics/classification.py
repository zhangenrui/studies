#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-30 1:54 下午

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

from sklearn.metrics._classification import (  # noqa
    accuracy_score as _accuracy_score
)

from huaytools.python.utils import get_logger

logger = get_logger()


def accuracy_count(y_true, y_pred) -> int:
    """
    accuracy 计数

    Examples:
        # 单标签分类
        >>> _y_true = [0,1,2,3]
        >>> _y_pred = [0,1,2,2]
        >>> accuracy_count(_y_true, _y_pred)
        3

        # 多标签分类：使用 one-hot 标签
        >>> _y_true = [[0,1],[1,1],[1,0],[0,0]]
        >>> _y_pred = [[0,1],[1,1],[1,0],[0,1]]
        >>> accuracy_count(_y_true, _y_pred)
        3

    """
    return int(_accuracy_score(y_true, y_pred, normalize=False))


def accuracy_score(y_true, y_pred, sample_weight=None, normalize=True) -> float:
    """@Metric Utils
    准确率计算

    Examples:
        # 单标签分类
        >>> _y_true = [0,1,2,3]
        >>> _y_pred = [0,1,2,2]
        >>> accuracy_score(_y_true, _y_pred)
        0.75
        >>> accuracy_score(_y_true, _y_pred, normalize=False)
        3
        >>> accuracy_score(_y_true, _y_pred, sample_weight=[1,2,3,4])
        0.6

        # 多标签分类：使用 one-hot 标签
        >>> _y_true = [[0,1],[1,1],[1,0],[0,0]]
        >>> _y_pred = [[0,1],[1,1],[1,0],[0,1]]
        >>> accuracy_score(_y_true, _y_pred)
        0.75

    Args:
        y_true: array-like of shape [N,] or [N, C]，其中 N 为样本数量，C 为标签数量
        y_pred: same shape as y_true
        sample_weight: 样本权重，array-like of shape (n_samples,)
        normalize: 为 `True` 时，相当于 np.average(y_true == y_pred, sample_weight)；
            为 `False` 时，相当于 np.dot(y_true == y_pred, sample_weight)
    """
    if sample_weight is None:
        sample_weight = [1] * np.shape(y_true)[0]  # 等权

    return _accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)


def accuracy_score_with_logits(y_true, y_pred):
    """"""


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()


