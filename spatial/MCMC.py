import scipy
import numpy as np
import pandas as pd
from scipy.stats import gamma, wishart, multivariate_normal, norm
import random

from utils.tests import *

def initialize_structures(adata, marker_genes, struct_key='struct', p_thresh=0.1):
    """
    The purpose of this function is to run a Fisher test to determine the prior location of
    the structure of interest through a Fisher Test. The following are inputs:
    1. adata: the annData object that holds the spatial transcriptomics sample
    2. marker_genes: a list of marker genes for the structure of interest
    3. struct_key: the key under which the label of each spot is stored in the adata object
    4. p_thresh: the threshold probability for whether a spot is marked as part of the structure
    from the Fisher test

    The following are returned:
    adata: the original annData object but annotated at each spot for whether it belongs to the structure
    or not
    """
    # Start by calculating z-scores of each spot regarding the genes expressed
    if scipy.sparse.issparse(adata.X):
        mat = pd.DataFrame.sparse.from_spmatrix(adata.X)
    else:
        mat = pd.DataFrame(adata.X)

    z_scores = scipy.stats.zscore(mat, axis=0)
    z_scores = np.where(np.isnan(z_scores), 0, z_scores)
    z_scores = pd.DataFrame(z_scores, index=adata.obs_names, columns=adata.var_names)

    # Prepare for the Fisher test
    gene_universe = list(adata.var_names)
    is_struct = []

    for i in range(len(adata.obs_names)):
        spot_genes = list(z_scores.iloc[i, :].sort_values(ascending=False)[0:100].index)
        _, p = FisherTest(spot_genes, marker_genes, gene_universe)
        if p < p_thresh:
            is_struct.append(1)
        else:
            is_struct.append(0)

    for i in range(len(is_struct)):
        is_struct[i] = str(is_struct[i])
    adata.obs[struct_key] = pd.Series(is_struct, dtype="category", copy=False)
    for i in range(len(adata.obs.follicle)):
        adata.obs[struct_key][i] = is_struct[i]

    return adata


def buildNeighborDict(adata, max_radius=6):
    """ 
    This function builds a neighbor dictionary for each spot that tells which other spots are directly
    adjacent to it. This is under the assumption that are working with Visium 10x datasets. 

    Inputs:
    1. adata: the annData object we are building the neighbor dictionary for
    2. max_radius: the radius under which we consider another spot a neighbor

    Output:
    neighbors: a dictionary object that takes in as key the label of a spot in the adata object and returns
    a list of other spots that are neighbors
    """

    # Begin by calculating a distance matrix
    spots = adata.obsm['spatial']
    dists_mat = scipy.spatial.distance.cdist(spots, spots)
    dists_mat = pd.DataFrame(dists_mat, index=adata.obs.index, columns=adata.obs.index)
    spots = pd.DataFrame(spots, index=adata.obs.index)

    # Initialize dictionary and sort through spots that are within the radius
    neighbors = {}
    for i, spot in enumerate(list(adata.obs.index)):
        sorted_vals = dists_mat.iloc[i, :].sort_values()
        sorted_vals = sorted_vals[sorted_vals > 0]
        sorted_vals = sorted_vals[sorted_vals < max_radius]
        spot_neighbors = list(sorted_vals.index)
        neighbors[spot] = spot_neighbors
    return neighbors


def PottsPrior(spot, adata, neighbors, struct_key='struct'):
    """
    This function calculates the Potts prior for each spot according to the structure labels
    and the neighbors. This is specifically for Visium datasets. 

    Inputs:
    1. spot: a string which represents the barcode for the spot we are calculating the prior for in 
    the adata object
    2. adata: an annData object that contains the labels for each spot being within the structure
    3. neighbors: a dictionary object that takes in as a key the spot of interest and returns a list 
    of other spots that are neighbors
    4. struct_key: a key that serves as the column name in adata.obs that tells whether spots belong to
    the structure or not
    """
    spot_neighbors = neighbors[spot]
    sum_val = 0

    # Loop through neighbors and determine contribution to sum
    for n_spot in spot_neighbors:
        if adata.obs[struct_key][spot] == adata.obs[struct_key][n_spot]:
            sum_val += 1
    return np.exp(3/len(spot_neighbors)*2*sum_val)


def returnTotalProduct(adata, flip_spot, mu_struct, mu_non_struct, lambda_mat, key):
    """
    This function returns the total product required for the MCMC procedure to determine whether a spot
    should be flipped in its label. 

    Inputs: 
    1. adata: the annData object that contains the spot labels
    2. flip_spot: the spot that is potentially going to have its label flipped
    3. mu_struct: the mean expression of the markers within the structure
    4. mu_non_struct: the mean expression of the markers outside of the structure
    5. lambda_mat: the matrix that describes the covariance between spots in and out of the structure
    6. key: the key for the labels in adata.obs

    Outputs:
    np.min((total_product, 1)): the acceptance probability for the potential flip
    """

    # Use a multivariate normal distribution to calculate the likelihoods of being in or out of the structure
    l1 = multivariate_normal.pdf(adata.obs.loc[flip_spot][['score not-struct', 'score struct']].to_numpy(), mu_struct, cov=np.linalg.inv(lambda_mat))
    l0 = multivariate_normal.pdf(adata.obs.loc[flip_spot][['score not-struct', 'score struct']].to_numpy(), mu_non_struct, cov=np.linalg.inv(lambda_mat))

    old_vals = adata.copy()

    # Calculate the total product when evaluating for the Potts Prior
    if adata.obs[key][flip_spot] == '1':
        # flipping struct to not struct
        likelihood_ratio = l0 / l1
        adata.obs[key][flip_spot] = '0'
        prior_ratio = PottsPrior(flip_spot, adata, adata.uns['neighbors']) / PottsPrior(flip_spot, old_vals, adata.uns['neighbors'])
        total_product = likelihood_ratio * prior_ratio
    else:
        # flipping not struct to struct
        likelihood_ratio = l1 / l0
        adata.obs[key][flip_spot] = '1'
        prior_ratio = PottsPrior(flip_spot, adata, adata.uns['neighbors']) / PottsPrior(flip_spot, old_vals, adata.uns['neighbors'])
        total_product = likelihood_ratio * prior_ratio

    # Return the acceptance probability
    return np.min((total_product, 1))


def GibbsUpdate(dim_reduction, labels, beta, n, alpha, d):
    """
    This function does Gibbs sampling to return the new parameters for the next step in MCMC

    Inputs:
    1. dim_reduction: a dataFrame that contains the signature gene expressions across each spot
    2. labels: the labels of each spot
    3. beta: the scaling value for the covariance update
    4. n: the number of total spots
    5. alpha: addition to Wishart distribution
    6. d: dimension of the covariance

    Outputs:
    1. mu_struct: The mean of the expression values of the signature across the structure
    2. mu_non_structure: The mean of the expression values of the signature across the non-struct
    3. lambda_mat: the covariance matrix for the Wishart distribution
    """

    # Update the means
    mu_struct = dim_reduction.loc[labels == '1', :].mean()
    mu_non_struct = dim_reduction.loc[labels == '0', :].mean()

    # Then update the covariance
    scale_mat = beta * np.eye(d)
    for i in range(n):
        y = dim_reduction.iloc[i, :]
        if labels[i] == '1':
            y = y - mu_struct
        else:
            y = y - mu_non_struct
        scale_mat = scale_mat + np.outer(y, y)
    lambda_mat = wishart.rvs(n + alpha, scale=np.linalg.inv(scale_mat))

    return mu_struct, mu_non_struct, lambda_mat


def MCMC(adata, num_steps, markers, key='struct', alpha=1, beta=0.1, p_thresh=0.1):
    """
    The full MCMC algorithm that finds the structures of interest

    Inputs:
    1. adata: the spatial tx sample that contains the dataset we want to find structures on
    2. num_steps: the number of steps we run MCMC for
    3. markers: the signature genes for the structure we are working with
    4. key: a string that acts as the accessor to whether a spot is in the struct or not
    5. alpha, beta: parameters for the Gibbs update
    6. p_thresh: the probability threshold for the Fisher test

    Outputs:
    1. adata: the annData now annotated with whether the spot is in the structure or not
    2. mu_struct: the mean expression of the signatures in the struct areas
    3. mu_non_struct: the mean expression of the signatures in the non-struct areas
    4. lambda_mat: the covariances used for the MCMC
    """
    # Some data parameters
    acceptances = 0
    n = len(adata.obs_names)

    # initialize the spots
    initialize_structures(adata, markers, struct_key=key, p_thresh=p_thresh)
    z = adata.obs[key]

    # select a random set of genes as markers of the non-spots
    gene_set = list(set(adata.var_names) - set(markers))
    non_markers = random.sample(gene_set, 100)

    # Work with a dimension reduction for the structure to initialize vals for MCMC
    signatures_follicle = pd.DataFrame(adata[:, markers].X.sum(axis=1), index=adata.obs_names)
    dim_reduction = pd.DataFrame(adata[:, non_markers].X.sum(axis=1), index=adata.obs_names)
    dim_reduction[1] = signatures_follicle[0]

    dim_reduction = dim_reduction.rename(columns={0: 'score not-struct', 1: 'score struct'})
    mu_struct = dim_reduction[adata.obs[key] == '1'].iloc[:, 0:2].mean().to_numpy()
    mu_non_struct = dim_reduction[adata.obs[key] == '0'].iloc[:, 0:2].mean().to_numpy()

    lambda_mat = beta * np.eye(2)

    # Build a dict for neighbors
    neighbors = buildNeighborDict(adata)

    # Begin MCMC procedure
    for i in range(num_steps):
        # If no steps possible
        if len(adata.obs[adata.obs[key] == '1'].index) < 1:
            break

        # First we do each of the spots via Metropolis Hastings
        u = random.uniform(0, 1)
        len_neighs = 0

        # Case of picking from structure neighborhood
        if u < 0.95:
            neighbors_list = []
            for spot in adata.obs[adata.obs[key] == '1'].index:
                spot_list = neighbors[spot].copy()
                spot_list.append(spot)
                neighbors_list.extend(spot_list)
            select_list = list(set(neighbors_list))

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

        old_vals = adata.copy()
        total_product = returnTotalProduct(adata, flip_spot, dim_reduction,
                                           mu_struct, mu_non_struct,
                                           lambda_mat, neighbors, key)

        # Check for the acceptance
        u = random.uniform(0, 1)
        if total_product >= u:
            accepted = True
            acceptances += 1
            z = adata.obs[key]
        else:
            accepted = False
            # Revert back to previous value
            adata.obs[key][flip_spot] = old_vals.obs[key][flip_spot]

        # Next we do the update of all the other parameters via Gibbs sampling
        if accepted:
            mu_struct, mu_non_struct, lambda_mat = GibbsUpdate(dim_reduction, adata.obs[key], 
                beta, n, alpha, d=2)

    return adata, mu_struct, mu_non_struct, lambda_mat
