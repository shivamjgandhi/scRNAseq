import pandas as pd
import scipy

def scoreCells(adata, genes_list, key):
    # Scores cells for expression from the genes_list and then appends the scores
    inter_genes = list(set(genes_list) & set(adata.var_names))
    if inter_genes:
        if scipy.sparse.issparse(adata.X):
            expr = pd.DataFrame.sparse.from_spmatrix(adata.X)
        else:
            expr = pd.DataFrame(adata.X)
        expr.columns = adata.var_names
        cell_means = expr.T.loc[inter_genes].mean()
        adata.obs[key] = cell_means.to_numpy()
    return adata

def cellTypeCounts(adata, cell_types, top_genes):
    for ct in cell_types:
        actual_genes = len(set(top_genes[ct]) & set(adata.var_names))
        print(ct + ': ' + str(actual_genes))