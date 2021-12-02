#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-20 7:21 下午

Author: huayang

Subject:

"""

from huaytools.nlp.bert.tokenization import _default_vocab_path

from transformers.models.bert import BertTokenizer

print(__file__)


def _test():
    """"""
    # doctest.testmod()

    vocab_file = _default_vocab_path

    t_tokenizer = BertTokenizer(vocab_file)
    t_tokenizer.batch_encode_plus




if __name__ == '__main__':
    """"""
    _test()
