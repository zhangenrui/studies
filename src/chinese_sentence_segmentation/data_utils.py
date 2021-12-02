#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-08 3:40 下午

Author: huayang

Subject:

"""
import re
import os
import sys
import json
import doctest

from huaytools.nlp.normalization import is_cjk

RE_W = re.compile(r'[,，.。!！\s～、~…]|$')
RE_NON_CN = re.compile(r'[^\u4E00-\u9FA5]')
RE_MULTI_comma = re.compile(r'\s+')
RE_MULTI_newline = re.compile(r'(\\\\n)+')


def normalize_for_sentence_segment(txt):
    """
    Examples:
        _txt = r'7月6日，我们拍婚纱照的日子，期待了好久\\n\\n在这里感谢ID的摄影师阿龙， \\n为我们忙碌了一天，从早上拍到下午5点多'
        _txt = normalize_for_sentence_segment(_txt)
        '7月6日，我们拍婚纱照的日子，期待了好久__##____##__在这里感谢ID的摄影师阿龙__##__为我们忙碌了一天，从早上拍到下午5点多'
    """
    txt = RE_MULTI_newline.sub('\n', txt)
    ps = txt.split('\n')

    nps = []
    for p in ps:
        cs = list(p)
        for idx, c in enumerate(cs):
            if not (is_cjk(c) or c.isalnum()):
                cs[idx] = ' '

        np = ''.join(cs).strip()
        np = RE_MULTI_comma.sub('，', np) + '，'
        nps.append(np)

    return '___'.join(nps)


def strip_non_cjk(s):
    start = 0
    for start, c in enumerate(s):
        if is_cjk(c) or c.isalnum():
            break

    end = 0
    for end, c in enumerate(s[::-1]):
        if is_cjk(c) or c.isalnum():
            break

    end = len(s) - end
    return s[start: end]


def _test():
    """"""
    doctest.testmod()

    txt = r'7月6日，我们拍婚纱照的日子，期待了好久\\n\\n在这里感谢ID的摄影师阿龙， 化妆师娇娇，助理小叶\\n为我们忙碌了一天，从早上拍到下午5点多'
    print(normalize_for_sentence_segment(txt))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    s = '我们拍婚纱照的日子，期待了好久'
    print(tokenizer.encode(s))
    s = '我们拍婚纱照的日子[MASK]期待了好久'
    print(tokenizer.encode(s))


if __name__ == '__main__':
    """"""
    _test()
