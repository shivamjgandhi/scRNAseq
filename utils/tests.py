from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

def FisherTest(gene_set1, gene_set2, gene_universe, alternative='greater'):
    """
    This function runs a fisher test on two gene sets
    :param gene_set1:
    :param gene_set2:
    :param gene_universe:
    :return:
    """
    plus_plus = len(list(set(gene_set1).intersection(gene_set2)))
    minus_plus = len(list(set(gene_set1) - set(gene_set2)))
    plus_minus = len(list(set(gene_set2) - set(gene_set1)))
    minus_minus = len(set(gene_universe) - set(gene_set1) - set(gene_set2))
    table = np.array([[plus_plus, plus_minus], [minus_plus, minus_minus]])
    oddsr, p = fisher_exact(table, alternative=alternative)
    return oddsr, p


def clusterVsDictFisher(adata, compare_dict, gene_universe):
    """
    This function takes in an adata object that contains clusters in adata.uns
    and then does a Fisher test with respect to the lists contained within compare_dict

    :param adata: an AnnData object, has clusters in adata.uns
    :param compare_dict: a comparison dict, elements should be lists
    :param gene_universe: the ambient universe as genes
    :return fisher_df: a dataframe consisting of the pvals from the Fisher test
    """
    num_clusters = len(adata.uns['rank_genes_groups']['scores'][0])
    fisher_df = pd.DataFrame(0.0, columns=np.arange(num_clusters), index=compare_dict.keys())
    for i in range(num_clusters):
        print('comparing cluster ' + str(i))
        gs1 = adata.uns['X' + str(i)]
        for key in compare_dict.keys():
            # Do the fisher test
            gs2 = compare_dict[key]
            _, p = FisherTest(gs1, gs2, gene_universe)
            fisher_df[i][key] = p

    return fisher_df


def BHCorrect(df, alpha=0.5):
    """
    Benjamini Hochberg correction of p-vals

    :param df: the dataframe of p-vals
    :param alpha: the alpha value of the test
    :return adjusted_p: the adjusted p-vals in a dataframe
    """
    pvals = df.stack()
    correct_ps = multipletests(pvals, method='fdr_bh', alpha=alpha)
    pvals[:] = correct_ps[1]
    adjusted_p = pvals.unstack()
    return adjusted_p