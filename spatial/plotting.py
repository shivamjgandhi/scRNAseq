import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import scipy
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, '..')
import os
cwd = os.getcwd()

from imageproc.images import returnCrossSection
from spatial.stats import scoreCells

# def cellTypePlot(annData, deconv, key):
#
#
# vals = quadrant['Adipose']
# dists = quadrant['dists']
# scaling = 100/vals.max()
# vals = vals*scaling
# expr_profile = []
# for i in range(len(dists)):
#     expr_profile.extend([dists[i]]*int(vals[i]))
# _, bins, _ = plt.hist(expr_profile, 20, density=1, alpha=0.5)
# mu, sigma = scipy.stats.norm.fit(expr_profile)
# best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# plt.plot(bins, best_fit_line)

def plotDistsOfType(deconv, key, cdata):
    cdata.obs.dists[cdata.obs.dists > 1.2] = 1.2
    dists = cdata.obs.dists
    vals = deconv[key]
    scaling = 100/vals.max()
    vals = vals*scaling
    expr_profile = []
    for i in range(len(dists)):
        expr_profile.extend([dists[i]]*int(vals[i]))
    _, bins, _ = plt.hist(expr_profile, 20, density=1, alpha=0.5)
    mu, sigma = scipy.stats.norm.fit(expr_profile)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, best_fit_line)
    plt.show()

def plotHullBdryPts(pts, y_kmeans, ind, bdrys, hulls):
    i = y_kmeans[ind]
    bdry = bdrys[i]
    max_hull = hulls[i]
    bdry_int = bdry.astype(np.uint8)
    bdry_int[bdry_int == 1] = 255

    im = Image.fromarray(bdry_int)
    img = ImageDraw.Draw(im)

    # Now draw both the line to the hull and the line to the membrane
    pt = pts.iloc[ind, :].to_numpy()
    bdry_pt, hull_v = returnCrossSection(pt, max_hull, bdry)
    bdry_pt = bdry_pt[::-1]
    hull_v = hull_v[::-1]
    pt = pt[::-1]
    max_hull = max_hull[:, ::-1]
    for i in range(len(max_hull)):
        start = max_hull[i]
        end = max_hull[(i + 1) % (len(max_hull))]
        shape = [tuple(start), tuple(end)]
        img.line(shape, fill=200, width=0)

    shape = [tuple(pt), tuple(bdry_pt)]
    img.line(shape, fill=200, width=0)
    shape = [tuple(pt), tuple(hull_v)]
    img.line(shape, fill=200, width=0)
    print(pt, bdry_pt, hull_v)

    im.show()


def plotSpecificPt(pt, bdry, max_hull):
    bdry_int = bdry.astype(np.uint8)
    bdry_int[bdry_int == 1] = 255

    im = Image.fromarray(bdry_int)
    img = ImageDraw.Draw(im)

    bdry_pt, hull_v = returnCrossSection(pt, max_hull, bdry)

    # need to get things in (x, y) form for drawing
    bdry_pt = bdry_pt[::-1]
    hull_v = hull_v[::-1]
    pt = pt[::-1]
    max_hull = max_hull[:, ::-1]
    for i in range(len(max_hull)):
        start = max_hull[i]
        end = max_hull[(i + 1) % (len(max_hull))]
        shape = [tuple(start), tuple(end)]
        img.line(shape, fill=200, width=0)

    shape = [tuple(pt), tuple(bdry_pt)]
    img.line(shape, fill=200, width=0)
    shape = [tuple(pt), tuple(hull_v)]
    img.line(shape, fill=200, width=0)
    print(pt, bdry_pt, hull_v)

    im.show()

def drawHullsBdrys(bdrys, hulls):
    total_bdry = bdrys[0]

    for i in range(1, len(bdrys)):
        total_bdry += bdrys[i]

    bdry_int = total_bdry.astype(np.uint8)
    bdry_int[total_bdry == 1] = 255

    im = Image.fromarray(bdry_int)
    img = ImageDraw.Draw(im)

    for j in range(len(hulls)):
        max_hull2 = hulls[j][:, ::-1]
        for i in range(len(max_hull2)):
            start = max_hull2[i]
            end = max_hull2[(i + 1) % (len(max_hull2))]
            shape = [tuple(start), tuple(end)]
            img.line(shape, fill=200, width=0)

    im.show()


def plotCellTypeSignatures(adata, signature_mat, img, spot_size, folder=None, img_key=None):
    cell_types = signature_mat.columns
    plt.rcParams["figure.figsize"] = (8, 8)
    cdata = adata.copy()
    for ct in cell_types:
        cdata = scoreCells(cdata, signature_mat[ct], ct)
        sc.pl.spatial(cdata, img_key=img_key, img=img, color=ct, size=1.5, spot_size=spot_size, save=ct + '.png')
        if folder:
            os.rename(cwd + '/figures/show' + ct + '.png', folder + '/' + ct + '.png')


def aBybPlot(a, b, adata_list, color,
             include_bars=True, include_labels=True, size=20,
             shared_legend=False, shared_title=True):
    fig, axs = plt.subplots(a, b, figsize=(size, size), constrained_layout=True)
    legends = []
    for i, adata in enumerate(adata_list):
        if (color in adata.var_names) or (color in adata.obs.columns):
            sc.pl.spatial(adata_list[i], img=adata_list[i].uns['image_lowres'], spot_size=4, color=color, ax=axs[int((i/b)), int((i % b))], show=False)
            if not include_bars:
                _lg = axs[int(i/b), int(i % b)].get_legend()
                if _lg:
                    legends.append(_lg)
                    _lg.remove()
                else:
                    fig.delaxes(fig.axes[-1])
            if not include_labels:
                axs[int(i/b), int(i % b)].set_xlabel('')
                axs[int(i/b), int(i % b)].set_ylabel('')
            if shared_title:
                old_title = axs[int(i/b), int(i % b)].get_title()
                axs[int(i/b), int(i % b)].set_title('')

    if shared_legend:
        leg = axs[0, 0].get_legend_handles_labels()
        fig.legend(leg[0], leg[1])

    if shared_title:
        if type(shared_title) == str:
            fig.suptitle(shared_title, fontsize=16)
        else:
            fig.suptitle(old_title, fontsize=16)
