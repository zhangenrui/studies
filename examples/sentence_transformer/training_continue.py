"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import os
import csv
import gzip
import math
import logging

from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from huaytools.python.custom import simple_argparse, BaseBunchDict

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',  # noqa
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class args:  # noqa
    """
    python training.py --model_name xx --train_batch_size 32 --num_epochs 3 --model_save_path xx
    """
    # model_name = 'bert-base-chinese'
    train_batch_size = 32
    num_epochs = 3
    model_save_path = None


args = simple_argparse(args)

if args.model_save_path is None:
    raise NotImplementedError('No Model')

# Load a pre-trained sentence transformer model
model = SentenceTransformer(args.model_save_path)

logging.info("Read train dataset")
train_samples = []
dev_samples = []
test_samples = []

# for row in rows:
#     score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
#     inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
#
#     if row['split'] == 'dev':
#         dev_samples.append(inp_example)
#     elif row['split'] == 'test':
#         test_samples.append(inp_example)
#     else:
#         train_samples.append(inp_example)

if not train_samples:
    raise NotImplementedError('No Train Data')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)  # noqa
train_loss = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')
warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=args.model_save_path)

# Load the stored model and evaluate its performance on STS benchmark dataset
# model = SentenceTransformer(args.model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='test')
# test_evaluator(model, output_path=args.model_save_path)
