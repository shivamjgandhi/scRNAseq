import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import SpatialDE
from imageproc.images import *
from imageproc.edges import *
from scipy.spatial import ConvexHull

def pipeline(adata, min_counts=50, min_cells=10, pct_counts_mt=20, n_top_genes=2000,
             img_key='hires', img=None, spot_size=33, method='wilcoxon', n_pcs=None, yes_spatial=False):
    """
    This function runs a spatial transcriptomics differential expression pipeline

    input: adata - an AnnData object. Should be processed so the image and spatial coordinates are
    included
    """
    # Deal with mitochondrial RNA and compute qc metrics
    adata.var['mt'] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Filter cells
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[1])

    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata = adata[adata.obs['pct_counts_mt'] < pct_counts_mt]
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalize
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes)

    # Manifold embedding and clustering
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='clusters')

    # Plot some of the covariates
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)

    # Visualize the clusters
    plt.rcParams["figure.figsize"] = (8, 8)
    if img_key is not None:
        sc.pl.spatial(adata, img_key=img_key, color='clusters', size=1.5, spot_size=spot_size)
    else:
        sc.pl.spatial(adata, img=img, color='clusters', size=1.5, spot_size=spot_size)

    # Compute the marker genes
    sc.tl.rank_genes_groups(adata, 'clusters', method=method)
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby='clusters')

    # Do spatial differential expression
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
    if yes_spatial:
        results = SpatialDE.run(coord, counts)
        adata.uns['spatial_DE_results'] = results

    return adata


def cellTypesPipeline(edata):
    full_img = edata.uns['image_hires']

    # Downsample, binarize, and smooth
    down_img = downSample(full_img)
    mult = returnBinary(down_img)
    kernel = np.ones((3, 3), np.float32) / 9
    for i in range(10):
        mult = cv2.filter2D(mult, -1, kernel)

    # Segment the image via kmeans
    pts = edata.obs[['y', 'x']].to_numpy(dtype='int')
    k_opt = returnOptimalK(pts)
    kmeanModel = KMeans(n_clusters=k_opt).fit(pts)
    y_kmeans = kmeanModel.predict(pts)
    cluster_centers = kmeanModel.cluster_centers_ / 10

    edata.obs['dists'] = 0
    # Now we work through each cluster
    for cluster in range(k_opt):
        print('beginning cluster ' + str(cluster))

        masked = maskImage(mult, cluster, cluster_centers)
        print('finished masking image')

        grown_masked = np.copy(masked)
        grown_masked = growTissue(grown_masked, num_runs=2)
        segmented_img = segmentImage(grown_masked)
        bdry = returnInnerBdry(segmented_img)
        print('finished getting inner membrane boundary')

        # Let's get a convex hull on it
        max_hull = returnConvexHull(masked)
        print('finished convex hull and inner boundary')

        # Get the pts in this cluster and apply the cross section algo to each point
        cluster_pts = edata.obs[['y', 'x']][y_kmeans == cluster] / 10
        for i in range(len(cluster_pts)):
            pt = cluster_pts.iloc[i, :].to_numpy()
            bdry_pt, hull_v = returnCrossSection(pt, max_hull, bdry)
            edata.obs.loc[cluster_pts.index[i], 'dists'] = computeDistance(pt, hull_v, bdry_pt)
        print('finished cluster ' + str(cluster))

    return edata

