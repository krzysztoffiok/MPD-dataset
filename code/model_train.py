import pandas as pd
import os
import argparse
import time
import statistics
import sys
import flair
import torch

# flair.device = torch.device('cuda')
# flair.device = torch.device('cpu')
# torch.cuda.empty_cache()

"""
Example use:

to train a single 10 fold CV language model:
python3 model_train.py --model=Glove --dataset=MPD

to train models for all final experiments from the paper:
bash ./grid_train

"""

parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--k_folds', required=False, type=int, default=10)
parser.add_argument('--epochs', required=False, type=int, default=1000)
parser.add_argument('--model', required=True, type=str, default="Glove",
                    help="choose language model type")
parser.add_argument('--dataset', required=True, type=str, default="MPD")
parser.add_argument('--block_print', required=False, default=False,
                    action='store_true', help='Block printable output')

args = parser.parse_args()

# number of folds for k_folds_validation
k_folds = args.k_folds
epochs = args.epochs
dataset = args.dataset
model_type = str(args.model)
model_type = model_type

path2 = []
results_path = "./results/{}/{}".format(dataset, model_type)

# disable printing out
block_print = args.block_print

if block_print:
    sys.stdout = open(os.devnull, 'w')

try:
    os.makedirs(results_path)
except FileExistsError:
    print("A test run with that name was already carried out. Try another name.")
    quit()

for i in range(k_folds):
    path2.append("./data/{}/model_subject_category_{}/".format(dataset, str(i)))
    try:
        os.mkdir('./data/{}/model_subject_category_{}'.format(dataset, str(i)))
    except FileExistsError:
        continue

time_schedule = []

for i in range(k_folds):

    """ section where we train our classifier """
    import flair.datasets
    from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, BertEmbeddings, ELMoEmbeddings
    from flair.embeddings import RoBERTaEmbeddings, XLNetEmbeddings, BytePairEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from pathlib import Path
    import logging
    from collections import defaultdict
    from torch.utils.data.sampler import Sampler
    import random, torch
    from flair.data import FlairDataset

    corpus: flair.data.Corpus = flair.datasets.ClassificationCorpus(Path(os.path.join(path2[i])),
                                                                    test_file='test_.tsv',
                                                                    dev_file='dev.tsv',
                                                                    train_file='train.tsv'
                                                                    )
    # way to select language model
    model_selector = {"Glove": [WordEmbeddings('glove')],
                      "FastText": [WordEmbeddings('en-news')],
                      "BPE": [BytePairEmbeddings('en')],
                      "FlairFast": [FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')],
                      "FlairNews": [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')],
                      "ElmoOriginal": [ELMoEmbeddings('original')],
                      'Bert': [BertEmbeddings('large-uncased')],
                      'BertLS': [BertEmbeddings(bert_model_or_path='bert-large-uncased',
                                                layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,"
                                                       "15,16,17,18,19,20,21,22,23,24", use_scalar_mix=True)],
                      "RoBERTa": [RoBERTaEmbeddings('roberta-base')],
                      "RoBERTaL": [RoBERTaEmbeddings('roberta-large')],
                      "RoBERTaLS": [RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-large",
                                                      layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,"
                                                             "15,16,17,18,19,20,21,22,23,24", use_scalar_mix=True)],
                      "XLNet": [XLNetEmbeddings(pretrained_model_name_or_path="xlnet-large-cased")],
                      "XLNetS": [XLNetEmbeddings(pretrained_model_name_or_path="xlnet-large-cased",
                                                 layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,"
                                                        "15,16,17,18,19,20,21,22,23,24", use_scalar_mix=True)],
                      "DistilBert": [BertEmbeddings('distilbert-base-uncased')]
                      }

    # Choose the tested embeddings
    word_embeddings = model_selector[model_type]

    # create an RNN
    document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                hidden_size=256,
                                                reproject_words=True,
                                                reproject_words_dimension=256,
                                                rnn_type="gru",
                                                bidirectional=False,
                                                rnn_layers=1)

    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(),
                                multi_label=False)

    # start counting training time before trainer
    time_schedule.append(time.perf_counter())

    # training parameters as chosen based on preliminary study
    trainer = ModelTrainer(classifier, corpus)
    trainer.train(base_path="{}".format(path2[i]),
                  max_epochs=epochs,
                  learning_rate=0.1,
                  mini_batch_size=8,
                  anneal_factor=0.5,
                  patience=20,
                  embeddings_storage_mode='gpu',
                  shuffle=True,
                  min_learning_rate=0.002,
                  )

    # add timestamp after trainer
    time_schedule.append(time.perf_counter())

    # rename the model files to fit model_type case
    os.rename(src="{}best-model.pt".format(path2[i]), dst="{}{}_best-model.pt".format(path2[i], model_type))
    os.remove("{}final-model.pt".format(path2[i]))

# compute k_fold training times
fold_times = []
for i in range(len(time_schedule)):
    if i % 2 == 0:
        try:
            fold_times.append(time_schedule[i+1]-time_schedule[i])
        except IndexError:
            print("end of range")
            fold_times.append(time_schedule[i])

aggregated_parameters = pd.DataFrame(data=fold_times, index=range(k_folds), columns=["Training time"])
aggregated_parameters.to_excel("{}/{}_timetrainingstats.xlsx".format(results_path, model_type))
