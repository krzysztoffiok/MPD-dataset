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

for preliminary experiment on MPD please use the bash script:
bash ./grid_train

"""

parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--k_folds', required=False, type=int, default=10)
parser.add_argument('--epochs', required=False, type=int, default=1000)
parser.add_argument('--model', required=False, type=str, default="subject_category",
                    help="sentiment or subject_category")
parser.add_argument('--test_run', required=True, type=str, default='p20lr.1mlr.0001a.5')
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


def decode_test_run(test_run):
    """
    Function to decode test_run name into sequence of training parameter values
    :param test_run:
    :return: patience, learning rate, minimal learning rate, anneal rate
    """
    x = test_run.split("lr.")
    patience = int(x[0][1:])
    lr = float(f"0.{x[1][:-1]}")
    y = x[2].split("a")
    mlr = float(f"0.{y[0]}")
    anneal_rate = float(f"0{y[1]}")
    print("Training parameters:\n"
          f"patience: {patience}\n"
          f"learning rate: {lr}\n"
          f"minimal learning rate: {mlr}\n"
          f"anneal rate: {anneal_rate}\n")
    return patience, lr, mlr, anneal_rate


try:
    patience, lr, mlr, anneal_rate = decode_test_run(test_run)
except:
    print("Incorrect test run provided.\n"
          "Appropriate format example: p20lr.1mlr.002a.5")


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
    from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings, FlairEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from pathlib import Path
    from collections import defaultdict
    from torch.utils.data.sampler import Sampler
    import random, torch

    from flair.data import FlairDataset

    corpus: flair.data.Corpus = flair.datasets.ClassificationCorpus(Path(os.path.join(path2[i])),
                                                                    test_file='test_.tsv',
                                                                    dev_file='dev.tsv',
                                                                    train_file='train.tsv'
                                                                    )

    # Choose the tested embeddings
    word_embeddings = [
                        WordEmbeddings('en-news')
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
                  learning_rate=lr,
                  mini_batch_size=8,
                  anneal_factor=anneal_rate,
                  patience=patience,
                  embeddings_storage_mode='gpu',
                  shuffle=True,
                  min_learning_rate=mlr,
                  )

    # add timestamp after trainer
    time_schedule.append(time.perf_counter())

    # rename the model files to fit test_run case
    os.rename(src="{}best-model.pt".format(path2[i]), dst="{}{}_best-model.pt".format(path2[i], test_run))
    os.remove("{}final-model.pt".format(path2[i]))  # , dst="{}{}_final-model.pt".format(path2[i], test_run))
    os.rename(src="{}training.log".format(path2[i]), dst="{}{}_training_{}.log".format(path2[i], test_run, i))

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
