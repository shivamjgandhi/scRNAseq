import scanpy as sc
import pandas as pd

def markersCSVs(AnnData, nameAccessor, name_df, cleaner=None):
    """
    Constructs a CSV of the marker genes and their expressions statistics

    :param AnnData: the AnnData object
    :param nameAccessor: the function that, given the dataframe of gene names, will
    return the correct gene name for the cds id
    :param name_df: the dataframe that matches gene name to id
    :param cleaner: a function that cleans the gene name
    :return csv_df of marker genes with expression stats:
    """
    stacked_genes = AnnData.uns['marker_genes'].T.stack()
    genes = AnnData.raw.var.fillna(0)
    # Create the df
    csv_df = pd.DataFrame({'Cluster': [], 'CDS ID': [], 'Gene Name': [],
                           'Alpha': [], 'Ref Alpha': [],
                           'Mean': [], 'Ref Mean': [],
                           'Mu': [], 'Ref Mu': []})
    delta = 0
    # For each cluster
    for i in range(len(AnnData.T)):
        # For each gene
        for j in range(20):
            cds_id = stacked_genes[i][j]
            alpha = genes.at[cds_id, 'alpha_' + str(i)]
            ref_alpha = genes.at[cds_id, 'ref_alpha_' + str(i)]
            mean = genes.at[cds_id, 'mean_' + str(i)]
            ref_mean = genes.at[cds_id, 'ref_mean_' + str(i)]
            mu = genes.at[cds_id, 'mu_' + str(i)]
            ref_mu = genes.at[cds_id, 'ref_mu_' + str(i)]
            if cleaner is not None:
                cds_id = cleaner(cds_id)
            gene_name = nameAccessor(name_df, cds_id)
            new_row = [i, cds_id, gene_name, alpha, ref_alpha, mean, ref_mean, mu, ref_mu]
            csv_df.loc[delta] = new_row
            delta += 1

    AnnData.uns['markers_csv'] = csv_df
    return csv_df

def intersectsCSVs(AnnData, thresh=0.05):
    """
    This function returns a csv of the intersection between clusters and pathways along
    with a p-val

    :param AnnData: an AnnData object
    :return csv_df: a dataframe with the information of interest
    """
    # Construct the csv
    csv_df = pd.DataFrame({'Cluster': [], 'Pathway': [], 'p_val': [], 'intersection': []})
    # Stack the pvals and filter everything below the threshold
    ps = AnnData.uns['corrected_p_vals']
    sig = ps.T.stack()[ps.T.stack() < thresh]
    # For each pathway that's significant, add in the row
    for j in range(len(sig)):
        cluster = sig.index[j][0]
        pathway = sig.index[j][1]
        p_val = sig.iloc[j]
        intersection = AnnData.uns['intersects'].at[pathway, cluster]
        new_row = [cluster, pathway, p_val, intersection]
        csv_df.loc[j] = new_row

    # Save this df to the AnnData object
    AnnData.uns['intersects_csv'] = csv_df
    return csv_df