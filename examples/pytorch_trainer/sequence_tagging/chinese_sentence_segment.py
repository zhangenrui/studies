#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-10 11:37 上午

Author: huayang

Subject:

"""
import re
import json
import random
import doctest

from itertools import islice

from tqdm import tqdm

from huaytools.nlp.normalization import is_cjk
from pytorch_trainer import Trainer, LossEvaluateCallback

BERT_MODEL_NAME = r'bert-base-chinese'

RE_W = re.compile(r'[,，.。!！\s～、~…]|$')
RE_NON_CN = re.compile(r'[^\u4E00-\u9FA5]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_MULTI_newline = re.compile(r'(\\\\n)+')
RE_BRACKETS = re.compile(r'\(.*?\)|（.*?）')


def normalize_for_one_sentence(s):
    """
    Examples:
        >>> normalize_for_one_sentence(r'上联：心里有座山，赵本山，怎么对下联？')
        '上联，心里有座山，赵本山，怎么对下联'
        >>> normalize_for_one_sentence(r'《复仇者联盟3》中哪个英雄表现的最出彩？')
        '复仇者联盟3，中哪个英雄表现的最出彩'
    """
    # 去除括号
    s = RE_BRACKETS.sub('', s)

    cs = list(s)
    for idx, c in enumerate(cs):
        # 把所有非文字、英文、数字替换为空格
        if not (is_cjk(c) or c.isalnum()):
            cs[idx] = ' '
    s = ''.join(cs).strip()

    # 把连续的空格替换为逗号
    s = RE_MULTI_SPACE.sub('，', s)
    return s


def gen_txt_diff():
    """把来自不同内容的短句拼接"""
    fw = open(r'data/chinese_sentence_segment_train_diff.txt', 'w', encoding='utf8')

    cnt = 0
    buf = []
    random_max = random.randint(50, 200)
    with open(r'/Users/huayang/workspace/my/data/short_news/train.json', encoding='utf8') as f:
        for ln in tqdm(f):
            it = json.loads(ln)
            s = it['text']
            s = normalize_for_one_sentence(s)
            buf.append(s)
            cnt += len(s)
            if cnt > random_max:
                txt = '。'.join(buf) + '。'
                fw.write(txt + '\n')
                cnt = 0
                buf = []
                random_max = random.randint(80, 200)


def gen_txt_same():
    """来自同一篇内容的句子"""
    # TODO


def get_data_loader(batch_size, device):
    """"""
    from huaytools.nlp.utils import split
    from huaytools.pytorch.data._basic import ToyDataLoader
    from transformers.models.bert.tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    rows = []
    with open(r'data/chinese_sentence_segment_train_diff.txt') as f:
        for ln in tqdm(islice(f, None)):
            txt = ' '.join(list(ln))
            _, tid = tokenizer.encode(txt, padding='max_length', truncation=True, max_length=256)
            txt_len = len(list(ln)) + 2
            mask = [1] * txt_len + [0] * max(0, 256 - txt_len)
            label = []
            for i in tid:
                if i == 8024:  # ，
                    label.append(1)
                elif i == 511:  # 。
                    label.append(2)
                else:
                    label.append(0)

            rows.append([tid, mask, label])

    rows_train, rows_val = split(rows, split_size=1000)

    train_data_loader = ToyDataLoader(rows_train, row2col=True, batch_size=batch_size, device=device)
    val_data_loader = ToyDataLoader(rows_val, row2col=True, batch_size=batch_size, device=device)

    return train_data_loader, val_data_loader


class MyTrainer(Trainer):
    """"""

    # 优化了默认 training_step 的逻辑，根据 batch 的类型推断 model 编码 batch 的方式，
    # 如果默认无法满足，需要重写 training_step，如下：
    def training_step(self, model, batch):
        """"""
        token_ids, token_masks, labels = batch
        probs, loss = model(token_ids, token_masks=token_ids, labels=labels)
        return probs, loss

    def set_model(self):
        """"""
        from huaytools.pytorch.nn import BertSequenceTagging
        bert_ckpt = r'/Users/huayang/workspace/my/studies/ckpt/bert-base-chinese'
        self.model = BertSequenceTagging(bert_ckpt, n_classes=2)  # 两种标签，逗号和句号

    def set_data_loader(self, batch_size, device):
        """
        准备训练数据

        原始数据：
            每一行文本

        """
        train_data_loader, val_data_loader = get_data_loader(batch_size, device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader


def _test():
    """"""
    doctest.testmod()


def main():
    """"""
    evaluate_callback = LossEvaluateCallback(early_stopping=True, baseline=0.2)
    '''
    这里因为有了训练经验，知道大概的 baseline，所以使用了 early_stopping，默认 early_stopping=False；
    LossEvaluateCallback 是 Trainer 默认使用的 Evaluator（evaluate_callback=None 时）
    '''
    trainer = MyTrainer(use_cpu_device=True, evaluate_callback=evaluate_callback)
    trainer.num_gradient_accumulation = 2
    trainer.batch_size = 32  # batch_size = 24
    trainer.num_evaluate_per_steps = 5
    trainer.num_train_epochs = 3

    trainer.train()


if __name__ == '__main__':
    """"""
    # _test()
    main()
