import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import argparse

"""
Example use:

python3 k_folds_prepare_data.py --k_folds=10

The script creates 10 directories with data for subject classification and also sentiment (unused).

The script creates a Monte Carlo Cross Validation split (the same test set for all train/val splits).
"""

parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--train_val_split', required=False, type=float, default=0.8)
parser.add_argument('--k_folds', required=False, type=int, default=10)

args = parser.parse_args()

# tran val split proportion
train_val_split = args.train_val_split

# number of folds to create
folds = args.k_folds

path = "./data/"

try:
    os.mkdir('./data/test')
except FileExistsError:
    None

# make the random division the same
np.random.seed(13)

data = pd.read_excel("{}mpd.xlsx".format(path)).sample(frac=1)
data = data[['row', 'sentiment', 'subject_category', 'sentence']]

data['sentiment'] = '__label__' + data['sentiment'].astype(str)
data['subject_category'] = '__label__' + data['subject_category'].astype(str)

# create k_folds from original dataset

data_folds = np.array_split(data, 10)
df_test = data_folds[0]
df_test = df_test.drop("row", axis=1)
df_test['sentiment'] = df_test["sentiment"].str.replace("__label__", "")
df_test['subject_category'] = df_test["subject_category"].str.replace("__label__", "")
df_test.to_excel(os.path.join("./data/test", "test.xlsx"))

# list of folder paths with folds
folds_path1 = []
folds_path2 = []

for i in range(folds):

    folds_path1.append('./data/model_sentiment_{}/'.format(str(i)))
    folds_path2.append('./data/model_subject_category_{}/'.format(str(i)))
    try:
        os.mkdir('./data/model_sentiment_{}'.format(str(i)))
    except FileExistsError:
        None  # continue
    try:
        os.mkdir('./data/model_subject_category_{}'.format(str(i)))
    except FileExistsError:
        None  # continue

    df = data

    # make test part the same for all models
    df_test = data_folds[0]

    # prepare the rest of data for train val split
    df_trainval = pd.concat([df, df_test])
    df_trainval = df_trainval.drop_duplicates(keep=False)

    # test split
    df_test = df_test.drop("row", axis=1)
    df_test_sent = df_test
    df_test_sent = df_test_sent.drop('subject_category', axis=1)
    df_test_sent.to_csv(os.path.join(folds_path1[i], "test_.tsv"), index=False, header=False, encoding='utf-8', sep='\t')
    df_test_subj = df_test
    df_test_subj = df_test_subj.drop("sentiment", axis=1)
    df_test_subj.to_csv(os.path.join(folds_path2[i], "test_.tsv"), index=False, header=False, encoding='utf-8', sep='\t')

    # train split
    df_train = df_trainval.sample(frac=train_val_split)

    # val split
    df_val = pd.concat([df_trainval, df_train])
    df_val = df_val.drop_duplicates(keep=False)
    df_val = df_val.drop("row", axis=1)

    df_val_sent = df_val
    df_val_sent = df_val_sent.drop('subject_category', axis=1)
    df_val_subj = df_val
    df_val_subj = df_val_subj.drop("sentiment", axis=1)

    df_val_sent.to_csv(os.path.join(folds_path1[i], "dev.tsv"), index=False, header=False, encoding='utf-8', sep='\t')
    df_val_subj.to_csv(os.path.join(folds_path2[i], "dev.tsv"), index=False, header=False, encoding='utf-8', sep='\t')

    df_train = df_train.drop("row", axis=1)
    df_train_sent = df_train
    df_train_sent = df_train_sent.drop('subject_category', axis=1)
    df_train_subj = df_train
    df_train_subj = df_train_subj.drop("sentiment", axis=1)

    df_train_sent.to_csv(os.path.join(folds_path1[i], "train.tsv"), index=False, header=False, encoding='utf-8', sep='\t')
    df_train_subj.to_csv(os.path.join(folds_path2[i], "train.tsv"), index=False, header=False, encoding='utf-8', sep='\t')