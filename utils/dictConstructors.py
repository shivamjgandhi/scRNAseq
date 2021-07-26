import scanpy as sc

def pathwayToGene(pathway_names, ko_to_path, GCF_to_K0, gene_df):
    """
    This function takes in a list of pathways for an experiment and returns the genes in
    three letter format is not None, else returns them in cds format

    :param pathway_names: gives the names of the pathways
    :param ko_to_path:
    :param GCF_to_K0:
    :param gene_df: a df that takes cds values and returns the three letter format
    :return pathway2g: a dict with the pathways as keys and the list of genes that make
    up the pathway in list format as the values
    """
    pathway2g = {}
    # Go through each pathway
    for j in range(len(pathway_names)):
        # Get the specific path
        pathway = pathway_names['map'][j]
        if pathway not in pathway2g.keys():
            # Get the K0 version of the gene ids
            gs2_pre = list(ko_to_path[ko_to_path['path'] == pathway]['K0'])
            gs2 = []
            for gid in gs2_pre:
                # Get a list of the cds associated with the K0 value
                cds = list(GCF_to_K0[GCF_to_K0['ko'] == gid]['id'])
                # In this case, we translate the cds id to the original
                if gene_df is not None:
                    for cd in cds:
                        gs2.append(gene_df[gene_df['id'] == cd]['gene'].item())
                else:
                    gs2.extend(cds)
            # And finally we put in the relevant genes for the pathway
            pathway2g[pathway] = gs2

    return pathway2g


def prefixToGene(gene_df, filter=0):
    """
    This function returns all the genes associated to a certain prefix

    :param gene_df: a df that takes cds values and returns the three letter format
    :param filter: an int, function filters out prefixes that have fewer than filter genes
    :return:
    """
    prefix2genes = {}
    for gene in list(gene_df.gene):
        # We do this to account for NaN values
        if type(gene) == str:
            if len(gene) == 4:
                # Get the prefix
                prefix = gene[0:3]
                if prefix in prefix2genes.keys():
                    prefix2genes[prefix].append(gene)
                else:
                    prefix2genes[prefix] = [gene]

    # Filter out prefixes with fewer than some number of genes
    prefix2genes_large = {}
    for key in prefix2genes.keys():
        if len(prefix2genes[key]) >= filter:
            prefix2genes_large[key] = prefix2genes[key]
    prefix2genes = prefix2genes_large

    return prefix2genes