import os
import json
import pandas as pd
import imageio
import numpy as np
import scanpy as sc

# Let's import and assemble Sathish's dataset
def constructAnnData(folder):
    # Get the annData object
    for _, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.h5'):
                a_file = file
    adata = sc.read_10x_h5(folder + '/' + a_file)
    adata.var_names_make_unique()

    f = open(folder + '/spatial/scalefactors_json.json')
    data = json.load(f)
    tissue_lowres_scalef = data['tissue_lowres_scalef']
    f.close()

    spatial_pos = pd.read_csv(folder + '/spatial/tissue_positions_list.csv', header=None)
    colnames = ['barcode', 'in_tissue', 'array_row', 'array_col', 'x', 'y']
    spatial_pos.columns = colnames
    valid_obs = [i in adata.obs_names for i in spatial_pos.barcode]
    spatial_pos = spatial_pos[valid_obs]
    positions_arr = np.zeros((len(spatial_pos), 2))
    for index, item in enumerate(adata.obs.index):
        ind = list(spatial_pos.barcode).index(item)
        positions_arr[index, :] = spatial_pos.iloc[ind, 4:6][::-1].to_numpy() * tissue_lowres_scalef

    adata.obsm['spatial'] = positions_arr

    img = imageio.imread(folder + '/spatial/tissue_lowres_image.png')
    adata.uns['image_lowres'] = img

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    return adata


# Returns the cumulative sum at each sample in a list
def Cumulative(lists):
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length + 1)]
    return cu_list[1:]


def sendToOriginalSpace(total_data, adata_list, new_key='total_clusters', old_key='clusters', categorical=True):
    """
    The purpose of this function is to take some value in total_data that's stored in obs and to 
    send that value back to the original adatas in adata_list. This is mostly for the purpose of plotting

    One requirement is that each adata in adata_list has a value called 'name' in the uns field
    that signifies which sample it is. 

    Inputs:
    total_data: this is a larger ann_data object that is composed of the annDatas in the adata_list
    adata_list: a list of the individual annDatas that compose the total_data annData
    new_key: the new key that stores the value in obs for the individual annDatas
    old_key: the old key from the total_data object
    """

    # First, we instantiate the values across all of the adatas
    sample_names = []
    if categorical:
        for adata in adata_list:
            adata.obs[new_key] = pd.Categorical(['0'] * adata.n_obs,
                total_data.obs[old_key].cat.categories,
                ordered=False)
            sample_names.append(adata.uns['name'])

    else:
        for adata in adata_list:
            adata.obs[new_key] = 0.0
            sample_names.append(adata.uns['name'])

    # Now go through each value in total_data
    for i in range(len(total_data.obs_names)):
        # Split the name of the spot in total_data
        sample = total_data.obs_names[i].split('.')[0]
        barcode = total_data.obs_names[i].split('.')[1] + '-1'

        # Select the right sample and put in the value
        adata_list[sample_names.index(sample)].obs.loc[barcode][new_key] = total_data.obs[old_key][i]

    return adata_list


# Mark the batch for each observation in total_data
def markBatch(total_data, adata_list, batch_key='batch'):

    total_data.obs[batch_key] = [0] * total_data.n_obs

    marks = [adata.n_obs for adata in adata_list]
    marks = Cumulative(marks)
    marks.insert(0, 0)

    j = 0
    for i in range(total_data.n_obs):
        if i >= marks[j + 1]:
            j += 1
            if j < len(marks):
                while marks[j] == marks[j + 1]:
                    j += 1
        total_data.obs[batch_key][i] = j

    return total_data

