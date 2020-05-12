import numpy as np
import pandas as pd
import time
import argparse
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, classification_report,\
    balanced_accuracy_score, roc_auc_score
from flair.models import TextClassifier
from flair.data import Sentence
import statistics
import sys
import os

"""
Example use:

to test classification of a single 10 fold CV language model:
python3 classification.py --k_folds=10 --subset=test --model=Glove --dataset=MPD

for preliminary experiment on MPD please use the bash script:
bash ./grid_class_lrp_mpd

for preliminary experiment on TREC6 please use the bash script:
bash ./grid_class_lrp_trec6

to carry out all final experiments from the paper:
bash ./grid_class
"""

""" Argument Parser """
parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--subset', required=False, type=str, default='test', help='string: which subset to analyze? train,'
                                                                               ' test, dev or whole file?')
parser.add_argument('--k_folds', required=False, type=int, default=10)
parser.add_argument('--block_print', required=False, default=False,
                    action='store_true', help='Block printable output')
parser.add_argument('--model', required=True, default='Glove')
parser.add_argument('--dataset', required=True, type=str, default="MPD")
args = parser.parse_args()

# number of folds for k_folds_validation
k_folds = args.k_folds

# file and subset to analyse
subset = args.subset
dataset = args.dataset

# group subject categories
if dataset == "MPD":
    gsc = ['subject', 'performance_assessment', 'recommendations', 'time', "technical"]
elif dataset == "TREC6":
    gsc = ['ABBR', 'DESC', 'ENTY', 'HUM', "LOC", "NUM"]

# test run descriptor
model_type = args.model

# disable printing out
block_print = args.block_print

if block_print:
    sys.stdout = open(os.devnull, 'w')

# define results path
results_path = "./results/{}/{}".format(dataset, model_type)

# read the data sentences
# choose the subset for analysis
if subset == 'test':
    data = pd.read_excel(f"./data/{dataset}/test/test.xlsx")

data = data[['subject_category', 'sentence']]
print('Data file to be analyzed loaded\n')

# placeholders for metrics
time_schedule = []
f1_list = []
mcc_list = []
f1sub_list = []
f1perf_list = []
f1rec_list = []
f1tec_list = []
f1time_list = []
f1ll_list = []
bas_list = []

# main loop
for fold in range(k_folds):
    # load pre-trained classifier
    try:
        subject_classifier = TextClassifier.load('./data/{}/model_subject_category_{}/{}_best-model.pt'.format(dataset,
                                                                                                               fold,
                                                                                                            model_type))
    except FileNotFoundError:
        print("----- Does such test run exist? Try another name. -----")
        quit()
    # placeholder for results of predictions
    flair_subject = []

    # add timestamp before all classification takes place
    time_schedule.append(time.perf_counter())

    # Main Analysis Loop:    start analysis
    for i in range(len(data)):
        # get the sentence to be analyzed
        sent = [str(data.iloc[i, 2])]

        for sent in sent:
            # print the sentence with original labels
            print('sentence: ' + sent, '| ' + '[' + str(data.iloc[i, 1]) + ']'
                  + ' | ' + '[' + str(data.iloc[i, 0]) + ']')

            # predict subject category
            sentence = Sentence(sent, use_tokenizer=True)
            subject_classifier.predict(sentence)
            sent_dict = sentence.to_dict()

            # print the results
            print('Subject category: ', sent_dict['labels'][0]['value'],
                  '{:.4f}'.format(sent_dict['labels'][0]['confidence']))
            group_subject = sent_dict['labels'][0]['value']

            # append the results
            flair_subject.append(group_subject)

    # add timestamp after all classification and before data aggregation
    time_schedule.append(time.perf_counter())

    # prepare DF for results
    pred_results = pd.DataFrame(columns=['flair_subject'])
    # add result columns to DF
    pred_results['flair_subject'] = flair_subject

    # add original data columns to results DF
    pred_results['label'] = data['subject_category'].values
    pred_results['sentence'] = data['sentence'].values

    """ compute metric scores """
    def metrics_compute(original_data, prediction, name):
        original_data = original_data
        prediction_list = prediction

        # compute global f1 score
        f1_results_score = f1_score(y_true=original_data, y_pred=prediction_list, average='weighted',
                                    labels=np.unique(prediction_list))

        output_string = 'F1 score for {} prediction: '.format(name), f1_results_score
        f1localscore = '{:.4f}'.format(f1_results_score)
        print(output_string)

        # prepare confusion matrix
        conf_matrix = confusion_matrix(y_true=original_data, y_pred=prediction_list)
        conf_matrix = pd.DataFrame(index=gsc, data=conf_matrix, columns=gsc)
        print(conf_matrix)

        # compute matthews correlation coefficient
        mcc = matthews_corrcoef(y_true=original_data, y_pred=prediction_list)
        print(mcc)

        # compute classification report which includes per-class F1 scores
        cr = classification_report(y_true=original_data, y_pred=prediction_list, target_names=gsc, output_dict=True)
        cr = pd.DataFrame(cr).transpose()
        print(cr)

        # compute class-balanced accuracy score
        bas = balanced_accuracy_score(y_true=original_data, y_pred=prediction_list)

        return f1localscore, conf_matrix, mcc, cr, bas

    # compute all metrics
    computed_metrics = metrics_compute(pred_results['label'].to_list(), pred_results['flair_subject'].to_list(),
                                       'Flair subject')

    # extract metrics
    f1_list.append(float(computed_metrics[0]))
    conf_matrix = computed_metrics[1]
    mcc_list.append(float(computed_metrics[2]))
    cr = computed_metrics[3]
    f1sub_list.append(float(cr.loc["subject"]["f1-score"]))
    f1perf_list.append(float(cr.loc["performance_assessment"]["f1-score"]))
    f1rec_list.append(float(cr.loc["recommendations"]["f1-score"]))
    f1tec_list.append(float(cr.loc["technical"]["f1-score"]))
    f1time_list.append(float(cr.loc["time"]["f1-score"]))
    bas_list.append(float(computed_metrics[4]))
    if model_type == "TREC6":
        f1ll_list.append(float(cr.loc[gsc[5]]["f1-score"]))

    conf_matrix.to_excel('{}/{}_summaryCM_{}_{}.xlsx'.format(results_path, model_type, fold, subset))
    cr.to_excel('{}/{}_summaryCR_{}_{}.xlsx'.format(results_path, model_type, fold, subset))

# compute avg and std k_fold classification times
fold_times = []
for i in range(len(time_schedule)):
    if i % 2 == 0:
        try:
            fold_times.append(time_schedule[i+1]-time_schedule[i])
        except IndexError:
            print("end of range")

# aggregate all class f1 score and mcc for all k_folds
aggregated_fold_scores = pd.DataFrame(data=f1_list, index=range(k_folds), columns=["F1 global"])
aggregated_fold_scores["MCC scores"] = mcc_list
aggregated_fold_scores["F1 subject"] = f1sub_list
aggregated_fold_scores["F1 performance assessment"] = f1perf_list
aggregated_fold_scores["F1 recommendations"] = f1rec_list
aggregated_fold_scores["F1 technical"] = f1tec_list
aggregated_fold_scores["F1 time"] = f1time_list
aggregated_fold_scores["Classification time"] = fold_times
aggregated_fold_scores["Balanced accuracy"] = bas_list
if model_type == "TREC6":
    aggregated_fold_scores[f"F1 {gsc[5]}"] = f1ll_list

# read time data from training of the model
training_data = pd.read_excel("{}/{}_timetrainingstats.xlsx".format(results_path, model_type))
training_data = training_data["Training time"].to_list()
aggregated_fold_scores["Training time"] = training_data[:k_folds]

if model_type == "MPD":
    parameter_list = [f1_list, mcc_list, f1sub_list, f1perf_list, f1rec_list, f1tec_list, f1time_list, fold_times, bas_list,
                  training_data]
elif model_type == "TREC6":
    parameter_list = [f1_list, mcc_list, f1sub_list, f1perf_list, f1rec_list, f1tec_list, f1time_list, f1ll_list,
                      fold_times, bas_list, training_data]

# compute mean and std for all metrics and add to DataFrame
mean_list = []
std_list = []
max_list = []
min_list = []
for i in range(len(parameter_list)):
    mean_list.append(statistics.mean(parameter_list[i]))
    try:
        std_list.append(statistics.stdev(parameter_list[i]))
    except statistics.StatisticsError:
        std_list.append(0)
    max_list.append(max(parameter_list[i]))
    min_list.append(min(parameter_list[i]))
aggregated_fold_scores.loc["mean"] = mean_list
aggregated_fold_scores.loc["std"] = std_list
aggregated_fold_scores.loc["max"] = max_list
aggregated_fold_scores.loc["min"] = min_list

# print to file all metrics
aggregated_fold_scores.to_excel("{}/{}_aggregated_fold_scores.xlsx".format(results_path, model_type))
