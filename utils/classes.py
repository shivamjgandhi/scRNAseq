from spatial.pipeline import *
from spatial.plotting import *
from spatial.stats import *
from spatial.cleaners import *
from spatial.MCMC import *

from utils.dataframe import *
from utils.pipeline import *
from utils.plotting import *
from utils.tests import *
from utils.cleaners import *


class sampleCollection():

    def __init__(self, sample_list, metadata=None):
        """
        Required inputs:
        sample_list: a list of samples

        Optional inputs:
        metadata: a pandas dataframe
        """
        # We have two options for the sample_list. The first is we pass in a list, the second is we 
        # pass in a string pointing to the folder with all the data
        self.unprocessed_samples = sample_list

        self.num_samples = len(sample_list)
        self.normalized = False
        self.total_data = None
        self.marks = self.setMarks()

        self.processed_samples = None
        self.processed = False
        self.metadata = metadata
        self.sample_info = {}


    def operateOnSamples(self, function, args=None, unprocessed=True, inplace=False):
        """
        Operates a function across each of the samples. Gives option to work on the unprocessed
        or processed samples and run function inplace or not in place
        """
        if inplace:
            for i, sample in enumerate(self.unprocessed_samples):
                if args == None:
                    function(sample)
                else:
                    input_args = {}
                    for arg in args:
                        input_args[arg] = self.sample_info[arg][i]
                    function(sample, input_args)

        if not inplace:
            results = []
            if unprocessed:
                samples = self.unprocessed_samples
            else:
                samples = self.processed_samples

            for i, sample in enumerate(samples):
                if args == None:
                    results.append(function(sample.copy()))
                else:
                    input_args = {}
                    for arg in args:
                        input_args[arg] = self.sample_info[arg][i]
                    results.append(function(sample.copy(), input_args))

            return results


    def preprocessData(self):
        """ 
        Does log and normalization of the unprocessed samples and then sets flag for normalization
        to true
        """
        for sample in self.unprocessed_samples:
            sc.pp.normalize_total(sample)
            sc.pp.log1p(sample)
        self.normalized = True


    def processSamples(self):
        """
        Runs each individual sample through the spatial pipeline with no spatialDE operation
        and then sets the flag to True
        """
        self.processed_samples = self.operateOnSamples(pipeline)
        self.processed = True


    def condenseData(self):
        """
        Takes all of the unprocessed samples and condenses them into one larger total_data AnnData
        object. This is for the purpose of clustering across the whole experiment
        """
        total_data = self.unprocessed_samples[0]
        for sample in self.unprocessed_samples[1:]:
            total_data = total_data.concatenate(sample)

        self.total_data = total_data


    def setMarks(self):
        """
        Sets the marks that separate the individual samples for the purpose of projecting
        back to the individual sampeles from the total_data object
        """

        def returnNObs(adata):
            return adata.n_obs

        marks = self.operateOnSamples(returnNObs)

        length = len(marks)
        cu_list = [sum(marks[0:x:1]) for x in range(0, length + 1)][1:]
        cu_list.insert(0, 0)

        self.marks = cu_list


    def projectToUnits(self, var, new_key):
        """
        Projects the variable from the total_data object back to the individual samples
        """

        def createCategorical(adata):
            adata.obs[new_key] = pd.Categorical(['0'] * adata.n_obs,
                                                self.total_data.obs[var].cat.categories,
                                                ordered=False)

        self.operateOnSamples(createCategorical, inplace=True)
        j = 0
        for i in range(self.total_data.n_obs):
            if i >= self.marks[j + 1]:
                j += 1
                if j < len(self.marks):
                    if self.marks[j] == self.marks[j + 1]:
                        j += 1
            self.unprocessed_samples[j].obs[new_key][i - self.marks[j]] = self.total_data.obs[var][i]


    def selfPermute(self, permutation):
        """
        Permutes the samples according to a given ordering. Note - does not operate on metadata.
        That will need to be permuted and set separately
        """
        self.unprocessed_samples = [self.unprocessed_samples[i] for i in permutation]
        if self.processed:
            self.processed_samples = [self.processed_samples[i] for i in permutation]

        for key in self.sample_info.keys():
            self.sample_info[key] = [self.sample_info[key][i] for i in permutation]

        # Recompute marks
        self.marks = self.setMarks()
        # Set total_data back to none
        self.total_data = None
