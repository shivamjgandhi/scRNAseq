import scanpy as sc
import pandas as pd
import seaborn as sns
import numpy as np
import os
import imageio
import json
import matplotlib.pyplot as plt
cwd = os.getcwd()

from spatial.pipeline import *
from spatial.plotting import *
from spatial.stats import *
from spatial.cleaners import *
from spatial.MCMC import *

from utils.dataframe import *
from utils.pipeline import *
from utils.plotting import *
from utils.tests import *
from utils.cleaners import *

# from Vy.sim import *
# from Lasso.classes import regression

def createAdataList(normalize=True):
    folder = './Sathish-Gut'
    sub_folders = [folder + '/' + name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

    adata_list_unprocessed = []
    for folder in sub_folders:
        adata_list_unprocessed.append(constructAnnData(folder))

    for i, adata in enumerate(adata_list_unprocessed):
        adata.uns['name'] = names[i].replace('-', '_')
        if normalize:
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata)

    return adata_list_unprocessed

def createRegressionObject():
    lig_rec = pd.read_csv('txts/ligand_receptor.literature.txt', sep='\t')
    expanded_cci_score = pd.read_csv('CSVs/expanded_ccis.csv', sep='\t', index_col=0)
    theta_expanded = pd.read_csv('CSVs/theta.tsv', sep='\t')

    theta = pd.read_csv('/Users/akumbhari/Documents/Smillie/spatial/BayesPrism/theta.merged.tsv',
                        sep='\t')
    tot_data = sc.read_h5ad('Assays/tot_data.h5ad')
    rowsums = np.array(tot_data.X.sum(axis=1))
    tot_data.X = 1e4*pd.DataFrame.sparse.from_spmatrix(tot_data.X).div(rowsums.squeeze(), axis=0).to_numpy()
    X_expanded = pd.read_csv('CSVs/X_expanded.csv', sep=' ')
    ct_sums = X_expanded.sum(axis=1)
    X_expanded = 1e4*X_expanded.div(ct_sums, axis=0)

    # Change the names of theta
    new_ind = [val.replace('-', '_') for val in theta.index]
    theta.index = new_ind
    theta_expanded.index = new_ind

    # Now sort everything based on theta since tot_data and theta share the same names
    i = sorted(theta.index)
    theta = theta.loc[i]
    theta_expanded = theta_expanded.loc[i]
    tot_data = tot_data[i, expanded_cci_score.index]
    
    main_regression = regression(tot_data, theta_expressor=theta_expanded,
                                    X_expressor=X_expanded,
                                    theta_contexts=theta, 
                                    ligand_receptors=lig_rec)
    
    return main_regression