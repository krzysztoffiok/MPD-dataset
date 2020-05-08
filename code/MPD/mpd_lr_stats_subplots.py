from scipy import stats
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os import listdir
from os.path import isfile, join
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.ticker import FuncFormatter
import argparse

"""

Example use:

python3 mpd_lr_stats_subplots.py --analyzed_set=mpd_f_tests --symbolic

options for different metrics and parameter testing results have to be configured in lines after 111.

"""

parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--analyzed_set', required=True, type=str, default="mpd_f_tests")
parser.add_argument('--symbolic', required=False, default=False, action='store_true')
parser.add_argument('--joint_plots', required=False, default=True, action='store_true')

args = parser.parse_args()

# number of folds for k_folds_validation
analyzed_set = args.analyzed_set
symbolic = args.symbolic
joint_plots = args.joint_plots
_fontsize = 20


class KFData(object):

    def __init__(self, file_list, columns, xlabel, natat):
        self.file_list = file_list
        self.columns = columns
        # self.title = title
        self.xlabel = xlabel
        self.natat = natat

    ylabel: str


test_run_list = ["p5lr.1mlr.0001a.5",
                   "p10lr.1mlr.0001a.5",
                   "p15lr.1mlr.0001a.5",
                   "p20lr.1mlr.0001a.5",
                   "p25lr.1mlr.0001a.5",
                   "p20lr.1mlr.001a.4",
                   "p20lr.2mlr.01a.5",
                   "p5lr.2mlr.008a.5",
                   "p10lr.05mlr.001a.5",
                   "p15lr.1mlr.001a.5",
                   "p10lr.3mlr.01a.3",
                   "p25lr.15mlr.001a.5",
                   "p20lr.1mlr.002a.5"]

file_list = [f"{x}_aggregated_fold_scores.xlsx" for x in test_run_list]

mpd_f_tests = KFData(file_list=file_list,
                    columns=test_run_list,
                    xlabel="MPD Learning rate parameters",
                    natat=True
                    )

mpd_lm = KFData(file_list=[
                               "BPE_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                               "glove_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                               "t2_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                               "flairnewsfast_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                               "flairnews_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                               "elmo_orig_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                               "BERTLunc_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                               "bertLunc_smix024_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                               "xlnet_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                               "xlnet_smix024_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                               "robertaBase_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                               "RobertaL_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                                "robertaLsmix024_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                                "distilbert_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                               ],
                   columns=[
                       "BPE",
                       "Glove",
                       "FastText",
                       "Flair Fast",
                       "Flair",
                       "Elmo",
                       "Bert",
                       "BertS",
                       "XLNet",
                       "XLNetS",
                       "Roberta",
                       "RobertaL",
                       "RobertaLS",
                       "DistilBert"
                            ],
                   xlabel="MPD",
                   natat=True
                   )

# define parameters
param_list = ["F1 global", "MCC scores", "Balanced accuracy", "Classification time", "Training time"]

# possible variants
selector = {"mpd_lm": mpd_lm, "mpd_tests": mpd_f_tests}
analyzed_set = selector[analyzed_set]

if analyzed_set == mpd_f_tests:
    param_list = [param_list[i] for i in [1, 4]]
    folder = "mpd_learning_rate_parameters_data"
    analyzed_set.file_list = [join(f"./results/{folder}/{x[:-28]}", x) for x in analyzed_set.file_list]
elif analyzed_set == mpd_lm:
    param_list = [param_list[i] for i in [1, 3, 4]]
    folder = "paper_xlsx_mpd"
    analyzed_set.file_list = [join(f"./results/{folder}", x) for x in analyzed_set.file_list]

analyzed_set.ylabel = param_list[0]
file_list = analyzed_set.file_list

# prepare data
df_compare = pd.DataFrame()

# read files
for num, file in enumerate(file_list):
    dftemp = pd.read_excel(file)
    for param in param_list:
        df_compare[f"{analyzed_set.columns[num]}_{param}"] = dftemp[param]

# drop statistical rows
df_compare = df_compare.drop([10, 11, 12, 13])

# convert training time from [s] to [h]
for column in df_compare.columns:
    if "Training time" in column:
        df_compare[column] = df_compare[column].apply(lambda x: x / 3600)

dtemp = pd.concat([df_compare.mean(), df_compare.std(), df_compare.max(), df_compare.median()], axis=1)
dtemp.columns = ["mean", "std", "max", "median"]
dtemp["text"] = dtemp["mean"].apply(lambda x: round(x, 3)).map(str) + "\u00B1" + \
                dtemp["std"].apply(lambda x: round(x, 3)).map(str)

# save to xlsx
if not analyzed_set.natat:
    dtemp.to_excel(f"./results/statistics/TREC6_{analyzed_set.xlabel}_mean.xlsx")
else:
    dtemp.to_excel(f"./results/statistics/MPD_{analyzed_set.xlabel}_mean.xlsx")

# Test Normality and save to .txt
# a) with scipy
# print(df_compare.apply(lambda x: pd.Series(stats.shapiro(x), index=['W', 'P'])).reset_index())

# # b) with pingouin
# if not analyzed_set.natat:
#     _normality = open(f"trec_{analyzed_set.xlabel}_normality.txt", "w")
# else:
#     _normality = open(f"natat_{analyzed_set.xlabel}_normality.txt", "w")
#
# print(pg.normality(df_compare), file=_normality)

# ANALYZE further. Prepare box plots
column_list = list(df_compare.columns)

if joint_plots:
    if symbolic:
        fig, ax = plt.subplots(1, len(param_list), figsize=(23, 10), gridspec_kw={'wspace': 0.15, 'hspace': 0}, dpi=200)
    else:
        fig, ax = plt.subplots(1, len(param_list), figsize=(23, 12), gridspec_kw={'wspace': 0.19, 'hspace': 0}, dpi=200) #, figsize=(15, 5))

    for num, param in enumerate(param_list[:]):
        print(param)

        # get columns with interesting parameter
        temp_col_list = [x for x in column_list if param in x]

        # optionally multiply some values
        if "MCC" in param:
            df_temp = df_compare[temp_col_list].multiply(1)
        else:
            df_temp = df_compare[temp_col_list]

        df_temp.columns = analyzed_set.columns

        # b) with pingouin
        # if not analyzed_set.natat:
        #     _normality = open(f"trec_{analyzed_set.xlabel}_{param}_normality.txt", "w")
        # else:
        #     _normality = open(f"natat_{analyzed_set.xlabel}_{param}_normality.txt", "w")

        # print(param, file=_normality)
        # print(pg.normality(df_temp), file=_normality)
        values = df_temp.median()

        # hack to make 3-line labels
        xlabels = []

        if analyzed_set == mpd_lm:
            for numm, k in enumerate(df_temp.columns):
                varr = 3
                if numm % varr == 0:
                    xlabels.append(f"{k}")
                elif numm % varr == 1:
                    xlabels.append(f"\n{k}")
                elif numm % varr == 2:
                    xlabels.append(f"\n\n{k}")

        else:
            for numm, k in enumerate(df_temp.columns):
                varr = 4
                if symbolic:
                    xlabels.append(numm)
                else:
                    if numm % varr == 0:
                        xlabels.append(f"{k}")
                    elif numm % varr == 1:
                        xlabels.append(f"\n{k}")
                    elif numm % varr == 2:
                        xlabels.append(f"\n\n{k}")
                    elif numm % varr == 3:
                        xlabels.append(f"\n\n\n{k}")

        print(xlabels)
        xi = list(range(1, len(xlabels)+1))

        # drow box plots
        fig.add_subplot(df_temp.boxplot(ax=ax[num]), sharey=False)
        plt.xticks(xi, xlabels)
        # plt.xticks(fontsize=_fontsize*1.5)

        # add text to plot
        for ii, v in enumerate(values):

            # remove leading 0
            if str(v).startswith("0"):
                v = str(v)[1:]
            ax[num].text(ii + 1.0, float(v), f"{v:.5}", ha="center", fontsize=_fontsize*0.75, fontweight="bold")

        # change titles of axis
        _title = param.replace("Classification time", "Inference Time [s]")
        _title = _title.replace("Training time", "Training Time [h]")

        # set axes title and size
        ax[num].set_title(_title, fontsize=_fontsize)

        # set global title for the whole figure
        fig.suptitle(analyzed_set.xlabel, fontsize=_fontsize*1.5)

        # set x label
        # ax[num].set_xlabel(analyzed_set.xlabel, fontsize=_fontsize)
        ax[num].tick_params(axis="x", labelsize=_fontsize*0.9, rotation=0)
        if symbolic:
            ax[num].tick_params(axis="x", labelsize=_fontsize*1, rotation=0)
        # set size of y axis ticks
        plt.yticks(fontsize=_fontsize)

        # remove leading 0 from yticks
        def my_formatter(x, pos):
            val_str = '{:g}'.format(x)
            if np.abs(x) > 0 and np.abs(x) < 1:
                return val_str.replace("0", "", 1)
            else:
                return val_str

        major_formatter = FuncFormatter(my_formatter)
        ax[num].yaxis.set_major_formatter(major_formatter)

        # ANOVA is computed here
        df_melt = pd.melt(df_temp.reset_index(), id_vars=["index"], value_vars=analyzed_set.columns)
        df_melt.columns = ["index", "treatments", "value"]
        model = ols('value ~ C(treatments)', data=df_melt).fit()
        # print(model.summary())
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

        if not analyzed_set.natat:
            tukey_ = open(f"./results/statistics/TREC6_{param}_{analyzed_set.xlabel}.txt", "w+")
        else:
            tukey_ = open(f"./results/statistics/MPD_{param}_{analyzed_set.xlabel}.txt", "w+")

        if len(file_list) > 2:
            m_comp = pairwise_tukeyhsd(endog=df_melt['value'], groups=df_melt['treatments'], alpha=0.05)
            print(f"{param}, NORMALITY:", file=tukey_)
            print(pg.normality(df_temp), file=tukey_)
            print("\n ANOVA:", file=tukey_)
            print(anova_table, file=tukey_)
            print("\n HSD Tukey:", file=tukey_)
            print(m_comp, file=tukey_)
            print(m_comp)
        elif len(file_list) < 3:
            print(f"{param}, NORMALITY:", file=tukey_)
            print(pg.normality(df_temp), file=tukey_)
            print("\n ANOVA:", file=tukey_)
            print(anova_table, file=tukey_)
            print("\n Wilcoxon:", file=tukey_)
            print(stats.wilcoxon(df_temp[analyzed_set.columns[0]], df_temp[analyzed_set.columns[1]]), file=tukey_)
            print(stats.wilcoxon(df_temp[analyzed_set.columns[0]], df_temp[analyzed_set.columns[1]]))

    plt.savefig(f"./results/images/{analyzed_set.xlabel}_{len(param_list)}.png")