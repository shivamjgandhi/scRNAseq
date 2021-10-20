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

    return adata


# Now that we've clustered, let's send it back to the original space
def sendToOriginalSpace(total_data, adata_list):
    marks = []
    for i, adata in enumerate(adata_list):
        marks.append(adata.n_obs)
        adata_list[i].obs['total_clusters'] = pd.Categorical(['0'] * adata.n_obs,
                                                             total_data.obs['clusters'].cat.categories,
                                                             ordered=False)

    def Cumulative(lists):
        length = len(lists)
        cu_list = [sum(lists[0:x:1]) for x in range(0, length + 1)]
        return cu_list[1:]

    marks = Cumulative(marks)
    marks.insert(0, 0)

    j = 0
    batch = []
    for i in range(total_data.n_obs):
        batch.append(j)
        if i >= marks[j + 1]:
            j += 1
        adata_list[j].obs['total_clusters'][i - marks[j]] = total_data.obs['clusters'][i]

    return adata_list