import scanpy as sc


class AnnEnsemble:
    """
    The purpose of this class is to keep all AnnData objects in
    """
    def __init__(self, AnnList):
        # AnnList is a list of AnnData Objects
        self.AnnDataList = AnnList

    def applyAll(self, func):
        # This function applies some function to all of the elements in an
        # AnnList
        return [func(x) for x in self.AnnList]
