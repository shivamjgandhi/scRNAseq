import scipy
import numpy as np
from scipy.stats import gamma, wishart, multivariate_normal, norm
import random

from utils.tests import *

def initialize_follicles(adata, follicle_genes=None):
    mat = np.array(scipy.sparse.csr_matrix.todense(adata.X))
    z_scores = scipy.stats.zscore(mat, axis=0)
    z_scores = np.where(np.isnan(z_scores), 0, z_scores)
    z_scores = pd.DataFrame(z_scores, index=adata.obs_names, columns=adata.var_names)

    top_genes_df = pd.DataFrame('', index=adata.obs_names, columns=np.arange(100))

    if follicle_genes == None:
        follicle_genes = ['CCL21', 'CCL19', 'CXCR4', 'CCR6', 'CD22', 'IL16', 'CD19', 'CD37']
    gene_universe = list(adata.var_names)
    is_follicle = []

    for i in range(len(top_genes_df.index)):
        spot_genes = list(z_scores.iloc[i, :].sort_values(ascending=False)[0:100].index)
        _, p = FisherTest(spot_genes, follicle_genes, gene_universe)
        if p < 0.1:
            is_follicle.append(1)
        else:
            is_follicle.append(0)

    for i in range(len(is_follicle)):
        is_follicle[i] = str(is_follicle[i])
    adata.obs['follicle'] = pd.Series(is_follicle, dtype="category", copy=False)
    for i in range(len(adata.obs.follicle)):
        adata.obs['follicle'][i] = is_follicle[i]

    return adata


# And now we try doing MCMC on these spots
# First get the neighbors for each
def buildNeighborDict(adata):
    spots = adata.obsm['spatial']
    dists_mat = scipy.spatial.distance.cdist(spots, spots)
    dists_mat = pd.DataFrame(dists_mat, index=adata.obs.index, columns=adata.obs.index)
    spots = pd.DataFrame(spots, index=adata.obs.index)
    neighbors = {}
    for i, spot in enumerate(list(adata.obs.index)):
        sorted_vals = dists_mat.iloc[i, :].sort_values()
        sorted_vals = sorted_vals[sorted_vals > 0]
        sorted_vals = sorted_vals[sorted_vals < 6]
        spot_neighbors = list(sorted_vals.index)
        neighbors[spot] = spot_neighbors
    return neighbors


def PottsPrior(spot, adata, neighbors):
    spot_neighbors = neighbors[spot]
    sum_val = 0
    for n_spot in spot_neighbors:
        if adata.obs['follicle'][spot] == adata.obs['follicle'][n_spot]:
            sum_val += 1
    return np.exp(3/len(spot_neighbors)*2*sum_val)


def returnTotalProduct(adata, flip_spot, mu_struct, mu_non_struct, lambda_mat, key):
    l1 = multivariate_normal.pdf(adata.obs.loc[flip_spot][['score not-follicle', 'score follicle']].to_numpy(), mu_struct, cov=np.linalg.inv(lambda_mat))
    l0 = multivariate_normal.pdf(adata.obs.loc[flip_spot][['score not-follicle', 'score follicle']].to_numpy(), mu_non_struct, cov=np.linalg.inv(lambda_mat))

    old_vals = adata.copy()
    if adata.obs[key][flip_spot] == '1':
        # flipping follicle to not follicle
        likelihood_ratio = l0 / l1
        adata.obs[key][flip_spot] = '0'
        prior_ratio = PottsPrior(flip_spot, adata, adata.uns['neighbors']) / PottsPrior(flip_spot, old_vals, adata.uns['neighbors'])
        total_product = likelihood_ratio * prior_ratio
    else:
        # flipping not follicle to follicle
        likelihood_ratio = l1 / l0
        adata.obs[key][flip_spot] = '1'
        prior_ratio = PottsPrior(flip_spot, adata, adata.uns['neighbors']) / PottsPrior(flip_spot, old_vals, adata.uns['neighbors'])
        total_product = likelihood_ratio * prior_ratio

    return np.min((total_product, 1))


def GibbsUpdate(dim_reduction, labels, beta, n, alpha, d):
    # Update the means
    mu_follicle = dim_reduction.loc[labels == '1', :].mean()
    mu_non_follicle = dim_reduction.loc[labels == '0', :].mean()

    # Then update the covariance
    scale_mat = beta * np.eye(d)
    for i in range(n):
        y = dim_reduction.iloc[i, :]
        if labels[i] == '1':
            y = y - mu_follicle
        else:
            y = y - mu_non_follicle
        scale_mat = scale_mat + np.outer(y, y)
    lambda_mat = wishart.rvs(n + alpha, scale=np.linalg.inv(scale_mat))

    return mu_follicle, mu_non_follicle, lambda_mat


def MCMC(adata, num_steps, marker_genes_df, key='follicle'):
    # Some data parameters
    acceptances = 0
    n = len(adata.obs_names)
    q = 2
    alpha = 1
    beta = 0.1

    # Initialize parameters to do MCMC on
    key = 'follicle'
    copy = adata.copy()
    z = adata.obs[key]

    # Build the dimension reduction necessary for the spots
    signatures_follicle = pd.DataFrame(adata[:, marker_genes_df[1]].X.sum(axis=1), index=adata.obs_names)
    dim_reduction = pd.DataFrame(adata[:, marker_genes_df[0]].X.sum(axis=1), index=adata.obs_names)
    dim_reduction[1] = signatures_follicle[0]

    dim_reduction = dim_reduction.rename(columns={0: 'score not-follicle', 1: 'score follicle'})

    mu_follicle = dim_reduction[copy.obs['follicle'] == '1'].iloc[:, 0:2].mean().to_numpy()
    mu_non_follicle = dim_reduction[copy.obs['follicle'] == '0'].iloc[:, 0:2].mean().to_numpy()

    lambda_mat = beta * np.eye(2)

    # Build a dict for neighbors
    neighbors = buildNeighborDict(adata)

    acceptances = 0
    for i in range(num_steps):
        if len(copy.obs[copy.obs['follicle'] == '1'].index) < 1:
            break
        # First we do each of the spots via Metropolis Hastings
        u = random.uniform(0, 1)
        # Case where we pick from follicle neighborhoods
        len_neighs = 0
        if u < 0.95:
            neighbors_list = []
            for spot in copy.obs[copy.obs['follicle'] == '1'].index:
                spot_list = neighbors[spot].copy()
                spot_list.append(spot)
                neighbors_list.extend(spot_list)
            select_list = list(set(neighbors_list))
            if len(neighbors_list) < 10:
                print(neighbors_list, i, len(copy.obs[copy.obs['follicle'] == '1'].index))

            while len_neighs == 0:
                j = random.randrange(len(select_list))
                flip_spot = select_list[j]
                len_neighs = len(neighbors[flip_spot])

        # Case where we pick at random
        else:
            while len_neighs == 0:
                j = random.randrange(len(z))
                flip_spot = adata.obs_names[j]
                len_neighs = len(neighbors[flip_spot])

        old_vals = copy.copy()
        total_product = returnTotalProduct(copy, flip_spot, dim_reduction,
                                           mu_follicle, mu_non_follicle,
                                           lambda_mat, neighbors, 'follicle')

        # Check for the acceptance
        u = random.uniform(0, 1)
        if total_product >= u:
            accepted = True
            acceptances += 1
            z = copy.obs[key]
        else:
            accepted = False
            # Revert back to previous value
            copy.obs[key][flip_spot] = old_vals.obs[key][flip_spot]

        # Next we do the update of all the other parameters via Gibbs sampling
        if accepted:
            mu_follicle, mu_non_follicle, lambda_mat = GibbsUpdate(dim_reduction, copy.obs['follicle'], beta, n, alpha,
                                                                   d=2)

    return copy, mu_follicle, mu_non_follicle, lambda_mat