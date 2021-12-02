#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-27 11:53
    
Author:
    huayang
    
Subject:
    Bert 原生分词器，移除了兼容 python2 的内容

References:
    https://github.com/google-research/bert/blob/master/tokenization.py
"""
import os
import doctest
from collections import OrderedDict

from typing import List, Callable, Sequence, Union

from huaytools.nlp.normalization import (
    is_cjk,
    is_whitespace,
    is_control,
    is_punctuation,
    remove_accents,
    convert_to_unicode
)

__all__ = [
    'BertTokenizer',
    'tokenizer',
]


def load_vocab(vocab_file, encoding='utf8'):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    index = 0
    with open(vocab_file, encoding=encoding) as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def split_by_whitespace(text):
    """Runs basic whitespace cleaning and splitting on a piece of text.

    Examples:
        >>> _text = '我爱python，我爱编程；I love python, I like programming.'
        >>> split_by_whitespace(_text)
        ['我爱python，我爱编程；I', 'love', 'python,', 'I', 'like', 'programming.']
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def split_by_punctuation(text):
    """Splits punctuation on a piece of text.

    Examples:
        >>> _text = '我爱python，我爱编程；I love python, I like programming.'
        >>> split_by_punctuation(_text)
        ['我爱python', '，', '我爱编程', '；', 'I love python', ',', ' I like programming', '.']
    """
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


class WordPieceTokenizer(object):
    """Runs WordPiece Tokenizer."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Examples:
            >>> _vocab = load_vocab(_default_vocab_path)
            >>> _tokenizer = WordPieceTokenizer(_vocab)
            >>> _tokenizer.tokenize('unaffable')
            ['u', '##na', '##ff', '##able']

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """
        # text = convert_to_unicode(text)

        output_tokens = []
        for token in split_by_whitespace(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BasicTokenizer(object):
    """"""

    def __init__(self, do_lower_case=True):
        """"""
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self.clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._add_space_around_cjk_chars(text)

        orig_tokens = split_by_whitespace(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = remove_accents(token)
            split_tokens.extend(split_by_punctuation(token))

        output_tokens = split_by_whitespace(" ".join(split_tokens))
        return output_tokens

    @staticmethod
    def clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _add_space_around_cjk_chars(text):
        """
        Examples:
            >>> _text = '我爱python，我爱编程；I love python, I like programming.'
            >>> BasicTokenizer._add_space_around_cjk_chars(_text)
            ' 我  爱 python， 我  爱  编  程 ；I love python, I like programming.'
        """
        output = []
        for char in text:
            cp = ord(char)
            if is_cjk(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class BertTokenizer(object):
    """@NLP Utils
    Bert 分词器

    Examples:
        >>> text = '我爱python，我爱编程；I love python, I like programming. Some unkword'

        # WordPiece 切分
        >>> tokens = tokenizer.tokenize(text)
        >>> len(tokens)
        22
        >>> assert [tokens[2], tokens[-2], tokens[-7]] == ['python', '##nk', 'program']

        # 模型输入
        >>> tokens, token_ids, token_type_ids = tokenizer.encode(text, return_token_type_ids=True)
        >>> tokens[:6]
        ['[CLS]', '我', '爱', 'python', '，', '我']
        >>> assert token_ids[:6] == [101, 2769, 4263, 9030, 8024, 2769]
        >>> assert token_type_ids == [0] * len(token_ids)

        # 句对模式
        >>> txt1 = '我爱python'
        >>> txt2 = '我爱编程'
        >>> tokens, token_ids, token_masks = tokenizer.encode(txt1, txt2, return_token_masks=True)
        >>> tokens
        ['[CLS]', '我', '爱', 'python', '[SEP]', '我', '爱', '编', '程', '[SEP]']
        >>> assert token_ids == [101, 2769, 4263, 9030, 102, 2769, 4263, 5356, 4923, 102]
        >>> assert token_masks == [1] * 10

        >>> # batch 模式
        >>> ss = ['我爱python', '深度学习', '机器学习']
        >>> token_ids = tokenizer.batch_encode(ss)
        >>> len(token_ids), len(token_ids[0])
        (3, 6)

    """

    token2id_map: dict  # {token: id}
    id2token_map: dict  # {id: token}

    def __init__(self, vocab_file,
                 do_lower_case=True,
                 token_cls='[CLS]',
                 token_sep='[SEP]',
                 token_unk='[UNK]',
                 token_mask='[MASK]',
                 token_pad='[PAD]',
                 verbose=0):
        self.token2id_map = load_vocab(vocab_file)
        self.id2token_map = {v: k for k, v in self.token2id_map.items()}
        if verbose > 0:
            print(f'Vocab size={len(self.token2id_map)}')
        # self.do_lower_case = do_lower_case
        self.basic_tokenizer = BasicTokenizer(do_lower_case)
        self.word_piece_tokenizer = WordPieceTokenizer(vocab=self.token2id_map)
        self.token_cls = token_cls
        self.token_sep = token_sep
        self.token_unk = token_unk
        self.token_mask = token_mask
        self.token_pad = token_pad
        self._padding_token_id = self.token2id_map[token_pad]

    def _basic_tokenize(self, text):
        """"""
        return self.basic_tokenizer.tokenize(text)

    def _word_piece_tokenize(self, text):
        return self.word_piece_tokenizer.tokenize(text)

    def tokenize(self, text):
        tokens = []

        if text:
            for token in self._basic_tokenize(text):
                for sub_token in self._word_piece_tokenize(token):
                    tokens.append(sub_token)

        return tokens

    def _encode(self, tokens1, tokens2, max_len):
        """"""
        tokens1, tokens2 = self._truncate(tokens1, tokens2, max_len)
        tokens, len_txt1, len_txt2 = self._concat(tokens1, tokens2)

        # 是否计算 token_type_id 和 token_mask，时间相差无几，故统一都计算，根据参数确定返回值
        token_id = self.convert_tokens_to_ids(tokens)
        token_type_id = [0] * len_txt1 + [1] * len_txt2
        token_mask = [1] * (len_txt1 + len_txt2)

        if max_len is not None:
            padding_len = max_len - len_txt1 - len_txt2
            token_id += [self._padding_token_id] * padding_len
            token_type_id += [0] * padding_len
            token_mask += [0] * padding_len

        return tokens, token_id, token_type_id, token_mask

    def encode(self, txt1, txt2=None,
               max_len=None,
               return_token_type_ids=False,
               return_token_masks=False,
               convert_fn=None):
        """
        Args:
            txt1:
            txt2:
            max_len:
            return_token_type_ids:
            return_token_masks:
            convert_fn:

        Returns: token_id, token_type_id, token_mask
        """

        tokens_txt1 = self.tokenize(txt1)
        tokens_txt2 = self.tokenize(txt2)

        tokens, token_id, token_type_id, token_mask = self._encode(tokens_txt1, tokens_txt2, max_len)

        if convert_fn is not None:
            token_id = convert_fn(token_id)
            token_type_id = convert_fn(token_type_id)
            token_mask = convert_fn(token_mask)

        inputs = [tokens, token_id]
        if return_token_type_ids:
            inputs.append(token_type_id)
        if return_token_masks:
            inputs.append(token_mask)
        return inputs if len(inputs) > 1 else inputs[0]

    def batch_encode(self,
                     texts: Union[List[str], List[List[str]]],
                     max_len: int = None,
                     return_token_type_ids=False,
                     return_token_masks=False,
                     convert_fn: Callable = None):
        """
        Args:
            texts:
            max_len:
            return_token_type_ids:
            return_token_masks:
            convert_fn: 常用的 `np.asarray`, `torch.as_tensor`, `tf.convert_to_tensor`

        Returns: token_ids, token_type_ids, token_masks
        """
        if isinstance(texts, str):
            texts = [texts]

        samples = []
        samples_length = []
        flag = 0  # 标记是否为 pair 输入
        for it in texts:
            if isinstance(it, str):
                txt1, txt2 = it, None
            elif isinstance(it, Sequence):
                txt1, txt2 = it[:2]
            else:
                raise TypeError(f'Unsupported sample type={type(it)}')

            tokens1 = self.tokenize(txt1)
            tokens2 = self.tokenize(txt2)
            flag = max(flag, len(tokens2))
            samples.append([tokens1, tokens2])
            samples_length.append(len(tokens1) + len(tokens2))

        if max_len is None:
            extra_len = 3 if flag > 0 else 2  # 特殊字符
            max_len = min(512, max(samples_length) + extra_len)

        token_ids = []
        token_type_ids = []
        token_masks = []
        for tokens1, tokens2 in samples:
            _, tid, sid, mask = self._encode(tokens1, tokens2, max_len=max_len)
            token_ids.append(tid)
            token_type_ids.append(sid)
            token_masks.append(mask)

        if convert_fn is not None:
            token_ids = convert_fn(token_ids)
            token_type_ids = convert_fn(token_type_ids)
            token_masks = convert_fn(token_masks)

        inputs = [token_ids]
        if return_token_type_ids:
            inputs.append(token_type_ids)
        if return_token_masks:
            inputs.append(token_masks)
        return inputs if len(inputs) > 1 else inputs[0]

    def convert_tokens_to_ids(self, tokens):
        return self._convert_by_vocab(self.token2id_map, tokens)

    def convert_ids_to_tokens(self, ids):
        return self._convert_by_vocab(self.id2token_map, ids)

    @staticmethod
    def _convert_by_vocab(vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def _concat(self, tokens_1st, tokens_2nd):
        packed_tokens_1st = [self.token_cls] + tokens_1st + [self.token_sep]
        if tokens_2nd:
            packed_tokens_2nd = tokens_2nd + [self.token_sep]
            return packed_tokens_1st + packed_tokens_2nd, len(packed_tokens_1st), len(packed_tokens_2nd)
        else:
            return packed_tokens_1st, len(packed_tokens_1st), 0

    @staticmethod
    def _truncate(tokens1, tokens2, max_len=None):
        """"""
        if max_len is None:
            max_len = len(tokens1) + len(tokens2)
            max_len += 3 if tokens2 else 2
            max_len = min(512, max_len)

        tokens1 = tokens1[:]
        tokens2 = tokens2[:]
        if tokens2:
            while True:
                total_len = len(tokens1) + len(tokens2)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(tokens1) > len(tokens2):
                    tokens1.pop()
                else:
                    tokens2.pop()
        else:
            del tokens1[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]

        return tokens1, tokens2


# 不是单例
# def get_tokenizer(vocab_file=None, **kwargs):
#     """
#
#     Args:
#         vocab_file:
#
#     Returns:
#
#     """
#     if vocab_file is None:
#         pwd = os.path.dirname(__file__)
#         vocab_file = os.path.join(pwd, '../data/vocab/vocab_21128.txt')
#
#     tokenizer = Tokenizer(vocab_file, **kwargs)
#     return tokenizer


# 模块内的变量默认为单例模式
_default_vocab_path = os.path.join(os.path.dirname(__file__), 'data_file/vocab_cn.txt')
tokenizer = BertTokenizer(_default_vocab_path)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
