#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-26 5:56 下午

Author: huayang

Subject: 句子分割

"""
import re
import doctest

SEPARATOR = r'@'
RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + SEPARATOR + r'(\w)', re.UNICODE)
UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + SEPARATOR + r'(\w)', re.UNICODE)


def replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def split_sentence(text, best=True):
    """@NLP Utils
    基于正则的分句

    Examples:
        >>> txt = '玄德幼时，与乡中小儿戏于树下。曰：“我为天子，当乘此车盖。”'
        >>> for s in split_sentence(txt):
        ...     print(s)
        玄德幼时，与乡中小儿戏于树下。
        曰：“我为天子，当乘此车盖。”

    References: https://github.com/hankcs/HanLP/blob/master/hanlp/utils/rules.py
    """

    text = re.sub(r'([。！？?])([^”’])', r"\1\n\2", text)
    text = re.sub(r'(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub(r'(…{2})([^”’])', r"\1\n\2", text)
    text = re.sub(r'([。！？?][”’])([^，。！？?])', r'\1\n\2', text)
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not best:
            yield chunk
            continue
        processed = replace_with_separator(chunk, SEPARATOR, [AB_SENIOR, AB_ACRONYM])
        for sentence in RE_SENTENCE.finditer(processed):
            sentence = replace_with_separator(sentence.group(), r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])
            yield sentence


def _test():
    """"""
    doctest.testmod()

    # import hanlp
    #
    # paragraph = '3.14 is pi. “你好！！！”——他说。劇場版「Fate/stay night [HF]」最終章公開カウントダウン！'
    # # ret = split_sentence(paragraph)
    # split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)
    # ret = split_sent(paragraph)
    # for s in ret:
    #     print(s)


if __name__ == '__main__':
    """"""
    _test()
