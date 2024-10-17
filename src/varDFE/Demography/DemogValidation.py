"""
Input validations for demographic models
"""

from dadi import Demographics1D
from varDFE.Demography import Demographics1D2
from dadi.DFE import DemogSelModels
from varDFE.DFE import DemogSelModels2

class DemogValidation():
    # define variables for splitting demographic parameters
    # TOUSERS: modify the upper and lower bounds to your needs
    def __init__(self):
        self.existing_models = {
            'one_epoch': [],
            'two_epoch': ["nua","Ta"],
            'three_epoch': ["nua", "nub", "Ta", "Tb"],
            'four_epoch': ["nua", "nub", "nuc", "Ta", "Tb", "Tc"]
        }

        self.upperbound = {
            'one_epoch': [],
            'two_epoch': [100,10],
            'three_epoch': [100,100,10,10],
            'four_epoch': [100,100,100,10,10,10]
        }

        self.lowerbound = {
            'one_epoch': [],
            'two_epoch': [1e-6,1e-6],
            'three_epoch': [1e-6,1e-6,1e-6,1e-6],
            'four_epoch': [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
        }

    def query_params(self, demog_model):
        upperbound = self.upperbound[demog_model]
        lowerbound = self.lowerbound[demog_model]
        return upperbound, lowerbound

    def ExistingModel(self, demog_model):
        """
        Return *demog_model* if existing demographic + selection or demographic model defined in this script, otherwise raise IOError.
        """

        if demog_model in self.existing_models.keys():
            return demog_model
        else:
            raise IOError("%s must specify a valid demography name" % demog_model)

    def split_demog_params(self, demog_params, demog_model):
        '''
        Split demographic upper bound, lower bound and initial values
        '''
        demog_params = list(map(float, demog_params.split(",")))

        # check that the demog_parameters has the same length
        param_names = self.existing_models[demog_model]
        if len(demog_params) == len(param_names):
            demog_paramdict = list(zip(param_names, demog_params, strict=True))
        else:
            raise IOError('Mismatching demog_params and demog_model')

        return demog_params, demog_paramdict

    def get_DFE_func_ex(self, demog_model):
        if demog_model == 'two_epoch':
            func_ex = DemogSelModels.two_epoch
        elif demog_model == 'three_epoch':
            func_ex = DemogSelModels2.three_epoch
        elif demog_model == 'four_epoch':
            func_ex = DemogSelModels2.four_epoch
        else:
            raise IOError('Wrong demog_model for DFE_func_ex')
        return func_ex

    def get_Demog_func_ex(self, demog_model):
        if demog_model == 'one_epoch':
            func_ex = Demographics1D.snm
        elif demog_model == 'two_epoch':
            func_ex = Demographics1D.two_epoch
        elif demog_model == 'three_epoch':
            func_ex = Demographics1D.three_epoch
        elif demog_model == 'four_epoch':
            func_ex = Demographics1D2.four_epoch
        else:
            raise IOError('Wrong demog_model for Demog_func_ex')
        return func_ex

