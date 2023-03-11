"""
Input validations for demographic models
"""

from dadi import Demographics1D
from varDFE.Demography import Demographics1D2
from dadi.DFE import DemogSelModels
from varDFE.DFE import DemogSelModels2

class DemogValidation():
    # define variables for splitting demographic parameters
    def __init__(self):
        self.existing_models = {
            'one_epoch': [],
            'two_epoch': ["nua","Ta"],
            'three_epoch': ["nua", "nub", "Ta", "Tb"],
            'four_epoch': ["nua", "nub", "nuc", "Ta", "Tb", "Tc"]
        }

        self.upperbound = {
            'one_epoch': [],
            'two_epoch': [10,5],
            'three_epoch': [10,10,5,5],
            'four_epoch': [10,10,10,5,5,5]
        }

        self.lowerbound = {
            'one_epoch': [],
            'two_epoch': [1e-6,1e-6],
            'three_epoch': [1e-6,1e-6,1e-6,1e-6],
            'four_epoch': [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
        }

    def query_params(self, modelname):
        upperbound = self.upperbound[modelname]
        lowerbound = self.lowerbound[modelname]
        return upperbound, lowerbound

    def ExistingModel(self, modelname):
        """
        Return *modelname* if existing demographic + selection or demographic model defined in this script, otherwise raise IOError.
        """

        if modelname in self.existing_models.keys():
            return modelname
        else:
            raise IOError("%s must specify a valid demography name" % modelname)

    def split_demog_params(self, demog_params, modelname):
        '''
        Split demographic upper bound, lower bound and initial values
        '''
        demog_params = list(map(float, demog_params.split(",")))

        # check that the demog_parameters has the same length
        param_names = self.existing_models[modelname]
        if len(demog_params) == len(param_names):
            demog_paramdict = list(zip(param_names, demog_params, strict=True))
        else:
            raise IOError('Mismatching demog_params and demog_model')

        return demog_params, demog_paramdict

    def get_DFE_func_ex(self, modelname):
        if modelname == 'two_epoch':
            func_ex = DemogSelModels.two_epoch
        elif modelname == 'three_epoch':
            func_ex = DemogSelModels2.three_epoch
        elif modelname == 'four_epoch':
            func_ex = DemogSelModels2.four_epoch
        else:
            raise IOError('Wrong demog_model for DFE_func_ex')
        return func_ex

    def get_Demog_func_ex(self, modelname):
        if modelname == 'one_epoch':
            func_ex = Demographics1D.snm
        elif modelname == 'two_epoch':
            func_ex = Demographics1D.two_epoch
        elif modelname == 'three_epoch':
            func_ex = Demographics1D.three_epoch
        elif modelname == 'four_epoch':
            func_ex = Demographics1D2.four_epoch
        else:
            raise IOError('Wrong demog_model for Demog_func_ex')
        return func_ex

