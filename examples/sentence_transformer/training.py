"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import os
import sys
import math
import logging

from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from huaytools.python.custom import simple_argparse, BaseBunchDict

logging.basicConfig(format='%(asctime)s - %(message)s',  # noqa
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class args:  # noqa
    """
    python training.py --model_name xx --train_batch_size 32 --num_epochs 3 --model_save_path xx
    """
    model_name = 'bert-base-chinese'
    train_batch_size = 32
    num_epochs = 3
    model_save_path = 'output/' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


args = simple_argparse(args)

word_embedding_model = models.Transformer(args.model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

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

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)  # noqa
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')

# Configure the training. We skip evaluation in this example
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
