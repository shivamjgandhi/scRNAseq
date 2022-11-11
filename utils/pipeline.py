import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as densify
import scipy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time


def expandedStats(adata, cluster_key='leiden'):
    """
    Appends expanded statistics to an AnnData object. Currently includes:
    1. alpha = percentage of cells expressing a gene
    2. Mean_expr = the mean expression of genes across all cells
    3. Mu_expr = the expression of genes across cells with non-zero expression

    :param adata: An AnnData object
    :return: AnnData object with expanded statistics
    """
    if scipy.sparse.issparse(adata.X):
        df = pd.DataFrame.sparse.from_spmatrix(adata.X)
    else:
        df = pd.DataFrame(adata.X)
    # We need to replace the index on the dataframe to match the original
    df.index = adata.obs.index
    num_clusters = len(adata.obs[cluster_key].cat.categories)
    for i in range(num_clusters):
        i = str(i)
        # This is the dataframe of cells that belong to cluster i
        sub_df = df[adata.obs[cluster_key] == i]
        # And the dataframe of cells not in i, the reference group
        ref_df = df[adata.obs[cluster_key] != i]
        adata.var['alpha_' + str(i)] = np.nan_to_num((sub_df > 0).mean())
        adata.var['ref_alpha_' + str(i)] = np.nan_to_num((ref_df > 0).mean().to_numpy())
        adata.var['mean_' + str(i)] = np.nan_to_num(sub_df.mean().to_numpy())
        adata.var['ref_mean_' + str(i)] = np.nan_to_num(ref_df.mean().to_numpy())
        adata.var['mu_' + str(i)] = np.nan_to_num(sub_df[sub_df > 0].mean().to_numpy())
        adata.var['ref_mu_' + str(i)] = np.nan_to_num(np.divide(ref_df.sum(), (ref_df > 0).sum()).to_numpy())

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
    adata = adata[:, adata.var.highly_variable]
    print('finished filtering based on highly variable genes')
    # Regress out effects of total counts per cell. Scale data to unit variance
    sc.pp.regress_out(adata, ['total_counts'])
    sc.pp.scale(adata, max_value=max_scale_value)
    print('finished regressing and scaling values')
    return adata


def geneSelection(adata, p_df, key='leiden', min_score=0, max_markers=None, cleaner=None, rankbyabs=False, ref_alpha=None):
    """
    This function selects the top marker genes for each cluster

    :param adata: AnnData object
    :param p_df: a dataframe containing the adjusted p-vals
    :param key: the cluster key from which you get the clusters
    :param min_score: The minimum score for a gene to be considered
    :param max_markers: the maximum number of markers
    :param cleaner: a cleaner function applied to gene names
    :return:
    """
    # Append the cluster keys
    clusters = list(adata.obs[key].cat.categories)
    key_names = []
    for clus in clusters:
        new_key = 'X' + str(clus)
        key_names.append(new_key)
        adata.uns[new_key] = None

    # Loop through each cluster and find which genes are relevant
    genes_list = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    scores = pd.DataFrame(adata.uns['rank_genes_groups']['scores'])
    if max_markers:
        marker_genes_df = pd.DataFrame(columns=clusters, index=np.arange(max_markers))
    else:
        marker_genes_df = pd.DataFrame(columns=clusters)

    if genes_list.shape[1] == 1:
        genes_list[clusters[1]] = genes_list[clusters[0]]
        scores[clusters[1]] = scores[clusters[0]]

    for clus in clusters:
        cluster_genes = genes_list[clus]
        sub_score = scores[clus]
        sub_score.index = cluster_genes

        if rankbyabs:
            sub_score = np.abs(sub_score).sort_values(ascending=False)

        sub_score = sub_score[sub_score > min_score]

        # Filter based on p-val
        p_val_genes = list(set((p_df[clus][p_df[clus] < 0.05]).index) & set(sub_score.index))
        sub_score = sub_score[p_val_genes]
        relevant_genes = sub_score.index

        if ref_alpha:
            alpha_genes = adata.var_names[adata.var['ref_alpha_' + str(clus)] > 0.05]
            relevant_genes = list(set(alpha_genes) & set(relevant_genes))

        # Filter out the maximum number of markers
        if max_markers is not None:
            relevant_genes = relevant_genes[0: max_markers]

        # Clean the genes if necessary
        if cleaner is not None:
            relevant_genes = [cleaner(g) for g in relevant_genes]

        adata.uns['X' + str(clus)] = relevant_genes
        marker_genes_df[clus][0:len(relevant_genes)] = relevant_genes

    return marker_genes_df


def computeMarkerGenes(adata, method='logreg', num_markers=None, ref_alpha=None,
                       min_score=0, cluster_key='leiden', name_cleaner=None,
                       rankby_abs=False):
    # Compute the marker genes from scanpy
    sc.tl.rank_genes_groups(adata, cluster_key, method=method, rankby_abs=rankby_abs)

    # Compute additional statistics
    adata = expandedStats(adata, cluster_key=cluster_key)

    # Compute p-vals
    p_df = computePVals(adata, method, key=cluster_key)

    # Select the appropriate genes and put them in the AnnData
    marker_genes_df = geneSelection(adata, p_df, key=cluster_key, min_score=min_score, max_markers=num_markers,
                                    cleaner=name_cleaner, ref_alpha=ref_alpha)

    return marker_genes_df


def checkpoint(t0):
    t1 = time.time()
    print('%s elapsed since last checkpoint' %(t1 - t0))
    return t1


def computePVals(adata, method='logreg', key='leiden', covs=[]):
    """
    This function computes the p-vals  from a differential expression test
    :param adata: an AnnData object
    :param method: the method by which we did the differential expression test
    :param key: the key for the clusters
    :return: a dataframe with the p-vals
    """
    num_genes = len(adata.var_names)
    clus_labels = adata.obs[key]
    clusters = list(set(clus_labels))
    num_clusters = len(clusters)
    if method == 'logreg':

        p_vals_adj = np.ones((num_genes, num_clusters))
        params = np.zeros((num_genes, num_clusters))
        if scipy.sparse.issparse(adata.X):
            d = adata.X.todense()
        else:
            d = adata.X
        d = np.array(d > 0)

        # Create an X matrix
        X = np.zeros((len(adata), 1 + len(covs)))
        for j, cov in enumerate(covs):
            X[:, j+1] = scipy.stats.zscore(np.log(adata.obs[cov]))

        for i, clus in enumerate(clusters):
            # Set an intercept
            X[:, 0] = (clus_labels == clus).astype(int)

            # Compute the adjusted p_val for the gene and cluster pair
            for l in range(num_genes):
                dl = d[:, l]
                logit_model = sm.Logit(dl, sm.add_constant(X))
                try:
                    result = logit_model.fit(disp=0)
                    p_vals_adj[l, i] = result.pvalues[1]
                    params[l, i] = result.params[1]
                except:
                    pass
        # Turn np array into a dataframe
        p_df = pd.DataFrame(p_vals_adj, index=adata.var.index[0:len(p_vals_adj)])
        p_df = p_df.fillna(1)
        params = pd.DataFrame(params, index=adata.var.index[0:len(p_df)])
        adata.uns['logreg coeffs'] = params
    elif method == 'wilcoxon':
        p_df = pd.DataFrame.from_records(adata.uns['rank_genes_groups']['pvals'])
        p_df.index = pd.DataFrame.from_records(adata.uns['rank_genes_groups']['names'])[clusters[0]]

    # Add the p-vals to the dataframe
    adata.uns['p-vals adj'] = p_df
    p_df.columns = clusters
    return p_df


# TODO
def run_scanpy(obj, counts=None, min_counts=5, min_genes=200, num_pcs=0, write_out=False,
               num_neighbors=100, DE_method='logreg', num_marker_genes=15):
    # Check input arguments

    # Preprocessing step
    adata = preprocessing(obj, min_cells=min_counts, min_genes=min_genes)

    # Do PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # Neighborbood graph: computing, embedding, and clustering
    sc.pp.neighbors(adata, n_neighbors=num_neighbors, n_pcs=num_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    # Find the marker genes
    marker_genes_df = computeMarkerGenes(adata, DE_method, num_markers=num_marker_genes)

    return marker_genes_df
