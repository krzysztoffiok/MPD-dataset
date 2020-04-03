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

"""

Example use:

python3 statsss_subplots_final.py

options for different metrics and parameter testing results have to be configured in lines 140-150.

"""


_fontsize = 20

class KFData(object):

    def __init__(self, file_list, columns, xlabel, natat):
        self.file_list = file_list
        self.columns = columns
        # self.title = title
        self.xlabel = xlabel
        self.natat = natat

    ylabel: str


trec_mini_batch_size = KFData(file_list=["t2_256_b4_p20_aggregated_fold_scores.xlsx",
                                         "t2_256_b8_p20_aggregated_fold_scores.xlsx",
                                         "t2_256_b32_p20_aggregated_fold_scores.xlsx",
                                         "t2_256_b64_p20_aggregated_fold_scores.xlsx"],
                              columns=["4", "8", "32", "64"],
                              xlabel="Mini Batch Size",
                              natat=False

                              )

trec_hidden_size = KFData(
    file_list=["t2_2_b32_p20_aggregated_fold_scores.xlsx", "t2_16_b32_p20_aggregated_fold_scores.xlsx",
               "t2_128_b32_p20_aggregated_fold_scores.xlsx", "t2_256_b32_p20_aggregated_fold_scores.xlsx",
               "t2_512_b32_p20_aggregated_fold_scores.xlsx"],
    columns=["2", "16", "128", "256", "512"],
    xlabel="Hidden Size",
    natat=False
    )

trec_shuffle = KFData(file_list=["t2_256_b32_p20_sf_aggregated_fold_scores.xlsx",
                                 "t2_256_b32_p20_aggregated_fold_scores.xlsx"],
                      columns=["False", "True"],
                      xlabel="Shuffling examples during training",
                      natat=False
                      )

trec_lm = KFData(file_list=[
                            "t5_256_b8_p20_aggregated_fold_scores.xlsx",
                            "t1_256_b8_p20_aggregated_fold_scores.xlsx",
                            "t2_256_b8_p20_aggregated_fold_scores.xlsx",
                            "flairfast_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                            "t6_256_b8_p20_aggregated_fold_scores.xlsx",
                            "elmoOrig_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                            "bertLunc_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                            "bertLuncsmix024_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                            "t7_256_b8_p20_aggregated_fold_scores.xlsx",
                            "xlnet_smix024_256_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                            "roberta_base_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                            "robertaL_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                            "roberta_large_smix024_b8_p20_mlr0002_aggregated_fold_scores.xlsx",
                            "distilbert_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx"
                            ],
                 columns=[
                          "BPE",
                          "Glove",
                          "FastText",
                          "Flair Fast",
                          "Flair",
                          "Elmo",
                          "Bert",
                          "Bert_smix",
                          "XLNet",
                          "XLNet_smix",
                          "Roberta",
                          "Roberta_L",
                          "Roberta_L_Smix",
                          "DistilBert"
                          ],
                 xlabel="TREC6",
                 natat=False
                 )

trec_f_tests = KFData(file_list=[
                                  "t2_256_b8_p5_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p10_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p15_repeat_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p20_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p25_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p20_mlr01_a04_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p20_lr02_mlr01_a05_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p5_lr02_mlr008_a05_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p10_lr005_mlr001_a05_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p15_mlr0001_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p10_lr03_mlr01_a03_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p25_lr015_mlr001_aggregated_fold_scores.xlsx",
                                  "t2_256_b8_p20_lr01_mlr0002_aggregated_fold_scores.xlsx",
                                  ],
                    columns=[
                           "p5lr.1mlr.0001a.5",
                           "p10lr.1mlr.0001a.5",
                           "p15lr.1mlr.0001a.5",
                           "p20lr.1mlr.0001a.5",
                           "p25lr.1mlr.0001a.5",
                           "p20lr.1mlr.001a.4",
                           "p20lr.2mlr.01a.5",
                           "p5lr.2mlr.008a.5",
                           "p10lr.05mlr.001a.5",
                           "p15l.1mlr.001a.5",
                           "p10lr.3mlr.01a.3",
                           "p25lr.15mlr.001a.5",
                           "p20lr.1mlr.002a.5",
                                ],
                    xlabel="Learning rate parameters",
                    natat=False
                    )

# define parameters
param_list = ["F1 global", "MCC scores", "Balanced accuracy", "Classification time", "Training time"]
# param_list = [param_list[1]]

# chose metrics for plots
param_list = [param_list[i] for i in [1, 3, 4]]
# param_list = [param_list[i] for i in [0, 2]]
joint_plots = True

##################### HERE DEFINE VARIANT #####################
analyzed_set = trec_lm
# analyzed_set = trec_hidden_size
# analyzed_set = trec_f_tests
###############################################################

if analyzed_set == trec_f_tests:
    folder = "learning_rate_parameters_data"
else:
    folder = "paper_xlsx"
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


def simple_plot(plot_param, stats_param):
    """
    plotting statistical parameters of selected model parameter (MCC, training time, inference etc.)
    :param stats_param: statistical parameter
    :param plot_param: model parameter like MCC or Training time
    :return:
    """
    temp_cols1 = [x for x in dtemp.index.to_list() if plot_param in x]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    values = dtemp.loc[temp_cols1][stats_param].to_list()
    labels = analyzed_set.columns

    plt.plot(range(len(labels)), values, 'bo', ms=15)  # Plotting data
    plt.xticks(range(len(labels)), labels)  # Redefining x-axis labels
    plt.yticks(fontsize=20)
    plt.title(plot_param + " " + stats_param)
    plt.tight_layout()

    # annotations near value points
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}\n{labels[i]}", ha="center", fontsize=_fontsize)


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

    fig, ax = plt.subplots(1, len(param_list), figsize=(20, 8), gridspec_kw={'wspace': 0.22, 'hspace': 0}, dpi=200) #, figsize=(15, 5))

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
        for numm, k in enumerate(df_temp.columns):
            # xlabels.append(k)

            if numm % 4 == 0:
                xlabels.append(f"{k}")
            elif numm % 4 == 1:
                xlabels.append(f"\n{k}")
            elif numm % 4 == 2:
                xlabels.append(f"\n\n{k}")
            elif numm % 4 == 3:
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
        ax[num].tick_params(axis="x", labelsize=_fontsize*0.6, rotation=0)

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

    # plt.legend()

    # plt.tight_layout()
    plt.savefig(f"./results/images/{analyzed_set.xlabel}_{len(param_list)}.png")

else:

    for num, param in enumerate(param_list[:]):
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=200)

        # get columns with interesting parameter
        temp_col_list = [x for x in column_list if param in x]

        # optionally multiply some values
        if "MCC" in param:
            df_temp = df_compare[temp_col_list].multiply(1)
        else:
            df_temp = df_compare[temp_col_list]

        df_temp.columns = analyzed_set.columns
        values = df_temp.median()

        # hack to make 2-line labels
        xlabels = []
        for numm, k in enumerate(df_temp.columns):
            if numm % 2 != 0:
                xlabels.append(f"{k}")
            else:
                xlabels.append(f"\n{k}")
        print(xlabels)
        xi = list(range(1, len(xlabels) + 1))

        # drow box plots
        fig.add_subplot(df_temp.boxplot(ax=ax), sharey=False)
        plt.xticks(xi, xlabels)

        # add text to plot
        for ii, v in enumerate(values):
            # remove leading 0
            if str(v).startswith("0"):
                v = str(v)[1:]
            ax.text(ii + 1.0, float(v), f"{v:.5}", ha="center", fontsize=_fontsize * 0.75, fontweight="bold")
        # plt.ylabel(param)
        _title = param.replace("Classification time", "Inference Time [s]")
        _title = _title.replace("Training time", "Training Time [h]")

        # plt.title(_title)
        ax.set_title(_title, fontsize=_fontsize)
        # plt.xlabel(analyzed_set.xlabel)
        ax.tick_params(labelsize=_fontsize)
        plt.yticks(fontsize=_fontsize)

        # remove leading 0 from yticks
        def my_formatter(x, pos):
            val_str = '{:g}'.format(x)
            if np.abs(x) > 0 and np.abs(x) < 1:
                return val_str.replace("0", "", 1)
            else:
                return val_str
        major_formatter = FuncFormatter(my_formatter)
        ax.yaxis.set_major_formatter(major_formatter)

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

            print(m_comp, file=tukey_)
            print(m_comp)
        elif len(file_list) < 3:
            print(stats.wilcoxon(df_temp[analyzed_set.columns[0]], df_temp[analyzed_set.columns[1]]), file=tukey_)
            print(stats.wilcoxon(df_temp[analyzed_set.columns[0]], df_temp[analyzed_set.columns[1]]))

    # plt.legend()

        # plt.tight_layout()
        plt.savefig(f"./results/images/{analyzed_set.xlabel}_{param}.png")
