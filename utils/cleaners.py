import scanpy as sc
import pandas as pd
import anndata2ri

def importSeurat(obj):
    anndata2ri.activate()

    sc.settings.verbosity = 3
    sc.logging.print_versions()

