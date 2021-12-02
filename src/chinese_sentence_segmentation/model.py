#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-08 2:44 下午

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

from huaytools.pytorch.nn import Bert, MaskedLanguageModel, build_bert_pretrained, tokenizer


class SegmentModel1(MaskedLanguageModel):
    """
    Model1: 直接使用 MLM 模型
    """

