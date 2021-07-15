import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from bioinfokit import analys, visuz



def volcanoPlot(df):
    visuz.gene_exp.volcano(df=df, lfc='log2FC', pv='p-value')
