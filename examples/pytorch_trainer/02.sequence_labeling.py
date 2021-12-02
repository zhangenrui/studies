#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-20 1:25 下午

Author: huayang

Subject:

"""

from typing import *

from itertools import islice

import torch
import torch.nn as nn

from torch import Tensor

from huaytools.pytorch.train.trainer import Trainer
from huaytools.pytorch.train.callback import Callback
from huaytools.pytorch.train.datasets import NerBertDatasets
from huaytools.nlp.ner_utils import ner_result_parse
from huaytools.nlp.bert_tokenizer import tokenizer


class MyTrainer(Trainer):

    def set_model(self):
        from huaytools.pytorch.nn import BertCRF
        self.model = BertCRF(n_classes=3)

    def training_step(self, model, batch) -> Union[Tensor, Tuple[Any, Tensor]]:
        token_ids, token_type_ids, masks, label_ids = batch
        probs, loss = model([token_ids, token_type_ids, masks], label_ids, masks)
        return probs, loss

    def set_data_loader(self, batch_size, device):
        # TODO: 解耦 args 和 NerBertDatasets
        args = self.args
        data = NerBertDatasets(args)
        args.id2label_map = data.id2label_map
        self.logger.info(data.id2label_map)
        self.train_data_loader = data.train_set
        self.val_data_loader = data.val_set


class ExampleCallback(Callback):
    """"""

    def on_after_optimize_step(self):
        T = self.trainer

        if not T.global_step % 3 == 0:
            return

        model = T.model
        batch = T.current_batch
        logger = T.logger

        token_ids, token_type_ids, masks, label_ids = batch
        prob, _ = model([token_ids, token_type_ids, masks], label_ids, masks)
        token_ids, mask = batch[0], batch[2]
        tags = model.decode(prob, mask)
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for tokens_id, m, ids in islice(zip(token_ids, mask, tags), 5):
            tokens_id = tokens_id[m.to(bool)].cpu().numpy().tolist()  # 移除 [PAD]
            ids = ids[: len(tokens_id)]
            tokens_id = tokens_id[1: -1]  # 移除 [CLS]、[SEP]
            ids = ids[1: -1]
            chunks = ner_result_parse(tokens_id, ids,
                                      token_id2name=tokenizer.id2token_map,
                                      label_id2name=T.args.id2label_map)
            tokens = tokenizer.convert_ids_to_tokens(tokens_id)
            # print(''.join(tokens), chunks)
            logger.info(f'\tseq={"".join(tokens)}, ret={chunks}')


def _test():
    """"""
    # doctest.testmod()
    # args = TrainConfig(src_train=r'data_files/ner_demo_100.txt', n_classes=3,
    #                    batch_size=8, val_percent=0.2, max_len=24, evaluate_per_step=3)
    trainer = MyTrainer()
    trainer.add_callback(ExampleCallback())
    trainer.train()


if __name__ == '__main__':
    """"""
    _test()
