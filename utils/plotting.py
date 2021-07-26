import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# def volcanoPlot(df):
#     visuz.gene_exp.volcano(df=df, lfc='log2FC', pv='p-value')

def heatMap(df, yticklabels=None, xlabel=None, ylabel=None, title=None,
            colorbar_label=None):
    """
    This function creates a heatmap from seaborn's package

    :param df: the dataframe we use to create a heatmap
    :param yticklabels: The labels we put on the y axis
    :param xlabel: The label for the x axis
    :param ylabel: The label for the y axis
    :param title: The title for the plot
    :param colorbar_label: The label for the colorbar
    :return ax: the heatmap
    """
    ax = sns.heatmap(df, yticklabels=yticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(colorbar_label)
    return ax

def pValHeatMap(ps, clip=None, filter=0.1, log_base=10, label_cleaner=None,
                cleaner_ref=None, xlabel=None, ylabel=None,
                title=None, colorbar_label=None):
    """
    This function returns a heatmap specifically for p-vals

    :param ps: p-values dataframe
    :param clip: The value we clip the p-vals at
    :param filter: The upper bound for p-vals we plot
    :param log_base: The base for the log transformation
    :param label_cleaner: The cleaner we apply to the labels on the heatmap
    :param cleaner_ref: The dataframe that the cleaner may take in
    :param xlabel: the xlabel for the map
    :param ylabel: the ylabel for the map
    :param title: the title for the heatmap
    :param colorbar_label: the label for the colorbar
    :return ax: the heatmap
    """
    # Only deal with p-vals below the filter
    ps = ps[ps.min(axis=1) < filter]
    # Clip p-vals
    if clip is not None:
        ps[ps < clip] = clip
        # log transform
    ps = -np.log10(ps)/np.log10(log_base)

    # Clean the labels
    if label_cleaner is not None:
        map_list = list(ps.index)
        if cleaner_ref is not None:
            labels = [label_cleaner(val, cleaner_ref) for val in map_list]
        else:
            labels = [label_cleaner(val) for val in map_list]
    else:
        labels = list(ps.index)

    # Plot
    ax = heatMap(ps, yticklabels=labels, xlabel=xlabel, ylabel=ylabel, title=title,
                 colorbar_label=colorbar_label)

    return ax

