import pandas as pd

df1 = pd.read_csv('marker_genes.txt')

df1.to_csv('marker_genes.csv', index=None)

