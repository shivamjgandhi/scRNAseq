import scanpy as sc
import pandas as pd
import os
import json
import numpy as np
import imageio

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def callR(function_path, func_name, obj, args):
    r = robjects.r
    r['source'](function_path)

    # Loading the function we have defined in R
    func = robjects.globalenv[func_name]

    # Convert it into r object for passing into r function
    with localconverter(robjects.default_converter + pandas2ri.converter):
        obj_r = robjects.conversion.py2rpy(obj)

    # Invoking the R function and getting the result
    df_result_r = func(obj_r, args)

    # Convert back into pd dataframe
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_from_r_df = robjects.conversion.rpy2py(df_result_r)

    return pd_from_r_df


def constructAnnData(folder):
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
