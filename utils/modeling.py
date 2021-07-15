from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix as densify
import numpy as np
import statsmodels.api as sm

def printvals():
    print('hello!')


#
# def LogisticPVals(X, y):
#     """
#     Returns the p-values from a logistic regression test done by scanpy
#
#     :param adata: AnnData object
#     :return:
#     """
#     # First we want to modify the scores so they're a matrix
#     # scores = adata.uns['rank_genes_groups']['scores']
#     # n = len(scores)
#     # m = len(scores[0])
#     # new_scores = np.zeros((n, m))
#     # for i in range(n):
#     #     for j in range(m):
#     #         new_scores[i, j] = scores[i][j]
#     #
#     # # Next we construct an array of cells by types
#     # n_cells = len(adata.obs['n_genes'])
#     # types = np.zeros((n_cells, m))
#     # for i in range(n_cells):
#     #     type = adata.obs['leiden'][i]
#     #     types[i, type] = 1
#     #
#     # # Finally, we do a regression from which we get p-values
#     # est = sm.Logit(types, densify(adata.raw.X).toarray())
#     # result = est.fit()
#     # print(result.summary())
#     model = LogisticRegression()
#     model.fit(X, y)
#     denom = (2.0*(1.0+np.cosh(model.decision_function(X))))
#     denom = np.tile(denom, (X.shape[1], 1)).T
#     F_ij = np.dot((X/deno))