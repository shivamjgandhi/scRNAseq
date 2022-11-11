import math
import os

# from cleaners import constructAnnData

def percentile(data, perc: int):
    size = len(data)
    return sorted(data)[int(math.ceil((size * perc) / 100)) - 1]


# def buildPath(folder):
#     sub_folders = [folder + '/' + name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
#     names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

#     adata_list_unprocessed = []
#     for folder in sub_folders:
#         adata_list_unprocessed.append(constructAnnData(folder))
    
#     return adata_list_unprocessed
