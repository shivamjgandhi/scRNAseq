import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as densify
import scipy
import pandas as pd
import numpy as np
import statsmodels.api as sm

# TODO
# def make_obj(counts=None, min_counts=10, min_genes=500):
#
#
# def differentialExpression(adata, clustering='leiden', group=None, method='logreg'):
#     if group is None:
#         categories = adata.obs[clustering].cat.categories
#         for cat in categories:
#             sc.tl.rank_genes_groups(adata, clustering, groups=[cat])
#     else:



def expandedStats(adata):
    """
    Appends expanded statistics to an AnnData object. Currently includes:
    1. Ref_alpha = percentage of cells expressing a gene
    2. Mean_expr = the mean expression of genes across all cells
    3. Mu_expr = the expression of genes across cells with non-zero expression

    :param adata: An AnnData object
    :return: AnnData object with expanded statistics
    """
    # Compute ref_alpha
    if scipy.sparse.issparse(adata.raw.X):
        df = pd.DataFrame.sparse.from_spmatrix(adata.raw.X)
    else:
        df = pd.DataFrame(adata.raw.X)
    ref_alpha = (df > 0).mean()
    adata.raw.var['ref_alpha'] = ref_alpha.to_numpy()
    # Compute mean
    adata.raw.var['mean_expr'] = df.mean().to_numpy()
    # Compute mu
    adata.raw.var['mu_expr'] = np.divide(df.sum(), (df > 0).sum()).to_numpy()

    return adata

def preprocessing(adata, normalize_total=True, target_sum=1e4, norm_method='log', min_mean=0.0125,
                  max_mean=3, min_disp=0.3, max_scale_value=10, min_genes=50, min_cells=4):
    """
    Function to run a preprocessing pipeline for single cell experiment analyses
    :param adata:
    :param normalize_total:
    :param target_sum:
    :return:
    """
    # First we do our basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print('finished filtering')

    # Next we compute quality control metrics based on mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    print('finished quality control')

    # Normalization
    if normalize_total:
        sc.pp.normalize_total(adata, target_sum)
    if norm_method == 'log':
        sc.pp.log1p(adata)
        print('Performed log normalization')
    print('finished normalization')

    # Filter based on highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
    print('Identified {} highly variable genes'.format(len(adata.var.highly_variable)))
    adata.raw = adata
    bdata = adata.copy()
    adata = adata[:, adata.var.highly_variable]
    print('finished filtering based on highly variable genes')
    # Regress out effects of total counts epr cell. Scale data to unit variance
    sc.pp.regress_out(adata, ['total_counts'])
    sc.pp.regress_out(bdata, ['total_counts'])
    sc.pp.scale(adata, max_value=max_scale_value)
    sc.pp.scale(bdata, max_value=max_scale_value)
    print('finished regressing and scaling values')
    return adata, bdata

def computePVals(adata, method='logreg'):
    n = len(adata.uns['rank_genes_groups']['scores'])
    m = len(adata.uns['rank_genes_groups']['scores'][0])
    if method == 'logreg':
        bdata = adata.raw.copy()
        y_general = [int(v) for v in adata.obs['leiden']]
        p_vals_adj = np.zeros((n, m))
        d = densify(bdata.X).toarray()
        for i in range(m):
            # Create a y vector
            y = np.zeros((len(y_general), 1))
            for j in range(len(y_general)):
                if y_general[j] == i:
                    y[j] = 1
            for l in range(n):
                model = sm.OLS(y, d[:, l])
                model2 = model.fit()
                p_vals_adj[l, i] = model2.pvalues
        p_df = pd.DataFrame(p_vals_adj)
        p_df.index = bdata.var.n_cells.index
    elif method == 'wilcoxon':
        p_vals_adj = np.zeros((n, m))
        ps = adata.uns['rank_genes_groups']['pvals_adj']
        for i in range(n):
            for j in range(m):
                p_vals_adj[i, j] = ps[i][j]
        p_df = pd.DataFrame(p_vals_adj)

    return p_df



# TODO
# def run_scanpy(name, obj=None, counts=None, min_counts=5, min_genes=200, num_pcs=0, write_out=False):
#     # Check input arguments
#
#     # Make singlecell object
#     if obj is not None:
#         obj = make_obj(counts=counts, min_counts=min_counts, min_genes=min_genes)
#
#     # Batch correction with variable genes
#
#     # Number of significant PCs
#
#     # Do PCA
#
#     # Fix duplicates
#
#     # Regress out PCs
#
#     # TSNE
#
#     # Cluster cells and run DE tests