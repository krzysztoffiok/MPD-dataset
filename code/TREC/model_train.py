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

python3 model_train.py --k_folds=10 --epochs=1 --test_run=t1
"""

parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--k_folds', required=False, type=int, default=10)
parser.add_argument('--epochs', required=False, type=int, default=1000)
parser.add_argument('--model', required=False, type=str, default="subject_category",
                    help="sentiment or subject_category")
parser.add_argument('--test_run', required=True, type=str, default='t1')
parser.add_argument('--block_print', required=False, default=False,
                    action='store_true', help='Block printable output')

args = parser.parse_args()

# number of folds for k_folds_validation
k_folds = args.k_folds
epochs = args.epochs
model_type = str(args.model)
test_run = args.test_run
path2 = []
results_path = "./results/{}".format(test_run)

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

    path2.append("./data/model_{}_{}/".format(model_type, str(i)))
    try:
        os.mkdir('./data/model_{}_{}'.format(model_type, str(i)))
    except FileExistsError:
        continue

time_schedule = []

for i in range(k_folds):

    """ section where we train our classifier """
    import flair.datasets
    from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, BertEmbeddings, ELMoEmbeddings
    from flair.embeddings import TransformerXLEmbeddings, OpenAIGPTEmbeddings, RoBERTaEmbeddings, OpenAIGPT2Embeddings
    from flair.embeddings import XLNetEmbeddings, BytePairEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from pathlib import Path
    import logging
    from collections import defaultdict
    from torch.utils.data.sampler import Sampler
    import random, torch

    from flair.data import FlairDataset
    # log = logging.getLogger("flair")

    corpus: flair.data.Corpus = flair.datasets.ClassificationCorpus(Path(os.path.join(path2[i])),
                                                                    test_file='test_.tsv',
                                                                    dev_file='dev.tsv',
                                                                    train_file='train.tsv'
                                                                    )

    # MORE OPTIONS, USE ONLY SOME LAYERS OF A MODEL AND HOW TO MIX PREDICTIONS FROM VARIOUS LAYERS
    # emb = RoBERTaEmbeddings(model="roberta.base", layers="0,1,2,3,4,5,6,7,8,9,10,11,12", pooling_operation="first",
    #                         use_scalar_mix=True)

    # Choose the tested embeddings by uncommenting
    word_embeddings = [#WordEmbeddings('glove'),
                           # WordEmbeddings('en-news'),
                         # WordEmbeddings('en-twitter'),
                         #  WordEmbeddings('en-crawl'),
                         # WordEmbeddings('en-extvec'),
         #FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')
        #                FlairEmbeddings('news-forward'),
        #                FlairEmbeddings('news-backward'),
        #                BertEmbeddings('bert-base-uncased'),
        # BertEmbeddings('bert-base-cased')
        #  BertEmbeddings(bert_model_or_path='bert-large-uncased', layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24", use_scalar_mix=True)
                     #  ELMoEmbeddings('small')
        # ELMoEmbeddings("original"),
        #                TransformerXLEmbeddings('transfo-xl-wt103')
        #                OpenAIGPTEmbeddings()
        # OpenAIGPT2Embeddings()
        # (pretrained_model_name_or_path="roberta-base", layers="-1", use_scalar_mix=False)
       #  XLNetEmbeddings(pretrained_model_name_or_path="xlnet-large-cased", layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24", use_scalar_mix=True)
RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-large", layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24", use_scalar_mix=True)
#         BytePairEmbeddings("en")
         ]

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

    # rename the model files to fit test_run case
    os.rename(src="{}best-model.pt".format(path2[i]), dst="{}{}_best-model.pt".format(path2[i], test_run))
    os.remove("{}final-model.pt".format(path2[i]))  # , dst="{}{}_final-model.pt".format(path2[i], test_run))

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
aggregated_parameters.to_excel("{}/{}_timetrainingstats.xlsx".format(results_path, test_run))
