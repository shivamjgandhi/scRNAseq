import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as densify
import scipy
import pandas as pd
import numpy as np
import statsmodels.api as sm

# # TODO
# def make_obj(counts=None, min_counts=10, min_genes=500):
#
#
# # def differentialExpression(adata, clustering='leiden', group=None, method='logreg'):
# #     if group is None:
# #         categories = adata.obs[clustering].cat.categories
# #         for cat in categories:
# #             sc.tl.rank_genes_groups(adata, clustering, groups=[cat])
# #     else:



def expandedStats(adata, cluster_key='leiden'):
    """
    Appends expanded statistics to an AnnData object. Currently includes:
    1. alpha = percentage of cells expressing a gene
    2. Mean_expr = the mean expression of genes across all cells
    3. Mu_expr = the expression of genes across cells with non-zero expression

    :param adata: An AnnData object
    :return: AnnData object with expanded statistics
    """
    # Compute alpha
    if scipy.sparse.issparse(adata.raw.X):
        df = pd.DataFrame.sparse.from_spmatrix(adata.raw.X)
    else:
        df = pd.DataFrame(adata.raw.X)
    # We need to replace the index on the dataframe to match the original
    df.index = adata.obs.index
    num_clusters = len(adata.obs[cluster_key].cat.categories)
    for i in range(num_clusters):
        # This is the dataframe of cells that belong to cluster i
        sub_df = df[adata.obs[cluster_key] == i]
        # And the dataframe of cells not in i, the reference group
        ref_df = df[adata.obs[cluster_key] != i]
        adata.raw.var['alpha_' + str(i)] = (sub_df > 0).mean().to_numpy()
        adata.raw.var['ref_alpha_' + str(i)] = (ref_df > 0).mean().to_numpy()
        adata.raw.var['mean_' + str(i)] = sub_df.mean().to_numpy()
        adata.raw.var['ref_mean_' + str(i)] = ref_df.mean().to_numpy()
        adata.raw.var['mu_' + str(i)] = sub_df[sub_df > 0].mean().to_numpy()
        adata.raw.var['ref_mu_' + str(i)] = np.divide(ref_df.sum(), (ref_df > 0).sum()).to_numpy()

    # We fill NaN values with 0 for simplicity
    adata.raw.var.fillna(0)
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


def geneSelection(adata, p_df, key='leiden', min_score=0, max_markers=None, cleaner=None):
    """
    This function selects the top marker genes for each cluster

    :param adata: AnnData object
    :param p_df: a dataframe containing the adjusted p-vals
    :param key: the cluster key from which you get the clusters
    :param min_score: The minimum score for a gene to be considered
    :param max_markers: the maximum number of markers
    :param cleaner: a cleaner functiona applied to gene names
    :return:
    """
    # Append the cluster keys
    num_clusters = len(adata.uns['rank_genes_groups']['scores'][0])
    cluster_names = [key]
    for i in range(num_clusters):
        new_key = 'X' + str(i)
        cluster_names.append(new_key)
        adata.uns[new_key] = None

    # Loop through each cluster and find which genes are relevant
    genes_list = pd.DataFrame.from_records(adata.uns['rank_genes_groups']['names'])
    scores = pd.DataFrame.from_records(adata.uns['rank_genes_groups']['scores'])
    for j in range(num_clusters):
        # All the genes associated with a cluster
        cluster_genes = genes_list[str(j)]
        relevant_genes = []
        over_score = True
        curr_gene = 0
        while over_score:
            # Retrieve the p_val, ref_alpha
            gene_name = cluster_genes[curr_gene]
            if (p_df[j][gene_name] < 0.05):
                # Add the gene to the relevant_gene list
                relevant_genes.append(gene_name)
            curr_gene += 1
            # In this case, we're below the minimum score so we can break the loop
            if scores[str(j)][curr_gene] < min_score:
                break
        # Filter out the maximum number of markers
        if max_markers is not None:
            relevant_genes = relevant_genes[0:max_markers]
        # Clean the genes if necessary
        if cleaner is not None:
            cleaned_genes = [cleaner(g) for g in relevant_genes]
            relevant_genes = cleaned_genes
        adata.uns['X' + str(j)] = relevant_genes

    # Create a dataframe for the top marker genes for each cluster
    marker_genes_df = pd.DataFrame(adata.uns['X0'])
    for i in range(1, num_clusters):
        marker_genes_df[i] = pd.DataFrame(adata.uns['X' + str(i)])

    return marker_genes_df


def computeMarkerGenes(adata, method='logreg', num_markers=None, ref_alpha=None,
                       min_score=0, cluster_key='leiden', name_cleaner=None):
    # Compute the marker genes from scanpy
    sc.tl.rank_genes_groups(adata, cluster_key, method=method)

    # Compute additional statistics
    adata = expandedStats(adata)

    # Compute p-vals
    p_df = computePVals(adata, method, key=cluster_key)

    # Select the appropriate genes and put them in the AnnData
    marker_genes_df = geneSelection(adata, p_df, key=cluster_key, min_score=min_score, max_markers=num_markers,
                                    cleaner=name_cleaner)

    return marker_genes_df


def computePVals(adata, method='logreg', key='leiden'):
    """
    This function computes the p-vals  from a differential expression test
    :param adata: an AnnData object
    :param method: the method by which we did the differential expression test
    :param key: the key for the clusters
    :return: a dataframe with the p-vals
    """
    num_genes = len(adata.uns['rank_genes_groups']['scores'])
    num_clusters = len(adata.uns['rank_genes_groups']['scores'][0])
    if method == 'logreg':
        bdata = adata.raw.copy()
        # This
        cluster_labels = [int(v) for v in adata.obs[key]]
        # Instantiate the p-vals-adjusted matrix
        p_vals_adj = np.zeros((num_genes, num_clusters))
        d = densify(bdata.X).toarray()
        for i in range(num_clusters):
            # Create a y vector
            print('check the code since you have a change')
            # y = (cluster_labels == i).astype(int)
            y = np.zeros((len(cluster_labels), 1))
            for j in range(len(cluster_labels)):
                if cluster_labels[j] == i:
                    y[j] = 1
            # Compute the adjusted p_val for the gene and cluster pair
            for l in range(num_genes):
                model = sm.OLS(y, d[:, l])
                model2 = model.fit()
                p_vals_adj[l, i] = model2.pvalues
        # Turn np array into a dataframe
        p_df = pd.DataFrame(p_vals_adj)
        p_df.index = bdata.var.index
    elif method == 'wilcoxon':
        p_vals_adj = np.zeros((num_genes, num_clusters))
        ps = adata.uns['rank_genes_groups']['pvals_adj']
        for i in range(num_genes):
            for j in range(num_clusters):
                p_vals_adj[i, j] = ps[i][j]
        p_df = pd.DataFrame(p_vals_adj)

    # Add the p-vals to the dataframe
    adata.uns['p-vals adj'] = p_df
    return p_df


# TODO
def run_scanpy(obj, counts=None, min_counts=5, min_genes=200, num_pcs=0, write_out=False,
               num_neighbors=100, DE_method='logreg', num_marker_genes=15):
    # Check input arguments

    # Preprocessing step
    val = preprocessing(obj, min_cells=min_counts, min_genes=min_genes)
    adata = val[0]
    bdata = val[1]

    # Do PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # Neighborbood graph: computing, embedding, and clustering
    sc.pp.neighbors(adata,n_neighbors=num_neighbors, n_pcs = num_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    # Find the marker genes
    marker_genes_df = computeMarkerGenes(adata, DE_method, num_markers=num_marker_genes)

    return marker_genes_df