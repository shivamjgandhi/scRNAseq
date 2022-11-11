import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import scipy


# def volcanoPlot(df):
#     visuz.gene_exp.volcano(df=df, lfc='log2FC', pv='p-value')

def heatMap(df, yticklabels='auto', xlabel=None, ylabel=None, title=None,
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


def quickHist(data, nbins=None):
    data = np.asarray(data)
    if not nbins:
        nbins = int(len(data)/100)
    x_min = np.min(data)
    x_max = np.max(data)
    plt.hist(data, bins=nbins, range=(x_min, x_max))
    plt.show()


def quickStats(data):
    quickHist(data)


def quickScatter(df, x, y):
    xvec = np.array(df[x])
    yvec = np.array(df[y])
    ax = sns.scatterplot(xvec, yvec)
    ax.set(xlabel=x, ylabel=y)
    return ax


def quickKMeans(model, pts):
    y_kmeans = model.predict(pts)
    # Create a scatter plot showing the spatial labels
    plt.scatter(pts[:, 0], pts[:, 1], c=y_kmeans, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


def kmeansSeparation(cluster_centers, mult):
    # Plot each of the cluster centers
    work = mult.copy()
    for i in range(len(cluster_centers)):
        work[int(cluster_centers[i, 1])-10: int(cluster_centers[i, 1])+10, int(cluster_centers[i, 0])-10:int(cluster_centers[i, 0])+10] = 20
    ones = np.where(work == 1)
    for i in range(len(ones[0])):
        pt = [ones[1][i], ones[0][i]]
        dists = np.linalg.norm(cluster_centers - pt, axis=1)
        clus = np.argmin(dists)
        work[pt[1], pt[0]] = 3*(clus + 1)
    plt.imshow(work)
    plt.show()


def scatterplot(x, y, col=None, lab=None, nlab=None, lab_size=12, xlab='', ylab='', font_size=14, w=10, h=8, title=None):
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from adjustText import adjust_text
    
    plt.rcParams['figure.figsize'] = [w, h]
    sns.set_style('white')
    sns.set_context('notebook', rc={'font.size':lab_size, 'axes.labelsize':font_size})    
    
    # fix input arguments
    x = np.array(x)
    y = np.array(y)
    if lab is not None:
        lab = np.array(lab, dtype=str)
    
    # select labels
    if lab is not None:
        if nlab is not None:
            l = np.copy(lab)
            l[:] = ''
            g = pd.cut(x, bins=10)
            for gi in g.unique():
                i = np.where(g == gi)[0]
                j = np.argsort(y[i])[-int(nlab/10):]
                l[i[j]] = lab[i[j]]
    
    # plot data
    plt.scatter(x, y, color=col)
    if lab is not None:
        text = [plt.text(x[i], y[i], l[i], ha='center', va='center') for i in range(len(x)) if l[i] != '']
        adjust_text(text)
    
    # format plot
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title:
        plt.title(title)

    sns.despine()


def volcanoPlot(select_clus, title=None, genes=None, fileName=None):
    # Create volc_df
    volc_df = pd.DataFrame(index = select_clus.var_names, columns=['-log10p', 'scores'])

    scores = pd.DataFrame(select_clus.uns['logreg coeffs'])
    scores.index = list(select_clus.var_names)
    volc_df['scores'] = np.clip(scores[0], -4, 4)
    volc_df['-log10p'] = np.clip(-1*np.log10(select_clus.uns['p-vals adj'].iloc[:, 0]), 0, 70)

    if genes:
        volc_df = volc_df.loc[genes, :]

    # Plot volc_df
    scatter = scatterplot(volc_df['scores'], volc_df['-log10p'], col='#cccccc', lab=volc_df.index, nlab=50, xlab='coefficient', ylab='-log10(p-value)', 
        title=title)
    
    if fileName:
        plt.savefig(fileName)

    plt.show()
    return scatter
