"""
Input validations for DFE PDF function forms
"""

from dadi.DFE import PDFs
from varDFE.DFE import PDFs2
import dadi

class PDFValidation():
    # define variables for the DFE functional forms
    def __init__(self):
        """
        Set initial parameters and names for each PDF
        """
        self.existing_pdfs = {
            'gamma': ['shape', 'scale'],
            'neugamma': ['pneu', 'shape', 'scale'],
            'neugammalet': ['plet', 'pneu', 'shape', 'scale'],
            'lognormal': ['mus', 'sigma'], # can't name as `mu`, will be confused with mutation rates
            'lourenco_eq': ['m', 'sigma', 'Ne', 'Ne_dadi']
        }

        self.pdf_function = {
            'gamma': PDFs.gamma,
            'neugamma': PDFs2.neugamma,
            'neugammalet': PDFs2.neugammalet,
            'lognormal': PDFs.lognormal,
            'lourenco_eq': PDFs2.lourenco_eq_pdf
        }

        self.Inference_optimizer = {
            'gamma': dadi.Inference.optimize_log,
            'neugamma': dadi.Inference.optimize_log,
            'neugammalet': dadi.Inference.optimize_log,
            'lognormal': dadi.Inference.optimize, # mu can be negative, therefore can't use optimize_log
            'lourenco_eq': dadi.Inference.optimize_log
        }

        # for information, outputted to the summary
        self.Spectra_integrate_methods = {
            'gamma': 'integrate',
            'neugamma': 'integrate',
            'neugammalet': 'integrate',
            'lognormal': 'integrate',
            'lourenco_eq': 'integrate_continuous_pos'
        }

        # upper bound, lower bound and params for each parameters
        # TODO: not sure if these are reasonable bounds
        self.upperbound = {
            'gamma': [2.0,1e+6], # upper bound at 10000 was limiting MM16 inference
            'neugamma': [1.0,2.0,1e+6],
            'neugammalet': [0.5,1.0,2.0,1e+6],
            'lognormal': [100.0,100.0],
            'lourenco_eq': [10.0,10.0,1e+9,1e+9],
        }

        # lower bound for each parameters
        self.lowerbound = {
            'gamma': [1e-3,1e-2],
            'neugamma': [1e-5,1e-3,1e-2], # 100 was too low for some species
            'neugammalet': [1e-5,1e-5,1e-3,1e-2],
            'lognormal': [-100,1e-5], # negative mean gives left skewed lognormal
            'lourenco_eq': [1e-5,1e-5,100,100]
        }

        # initial values for each parameters
        self.initval = {
            'gamma': [0.2,4000.0], # usual values output
            'neugamma': [0.3,0.2,4000.0],
            'neugammalet': [0.001,0.3,0.2,4000.0], # didn't use EW's initival values [0.2,0.999,0.1,2000.0] 0.2+0.9=1.1 reduced all gammas
            'lognormal': [1.0,0.1], # most values were positive, start from positive
            'lourenco_eq': [0.5,0.1,2400.0,4000.0] # near Huber 2017 human data
        }

    def query_params(self, pdfname):
        upperbound = self.upperbound[pdfname]
        lowerbound = self.lowerbound[pdfname]
        initval = self.initval[pdfname]
        return upperbound, lowerbound, initval

    def ExistingPDF(self, pdfname):
        """
        Return *pdfname* if existing DFE pdf has been defined in this script, otherwise raise IOError.
        pdfname: input PDF name
        """

        if pdfname in self.existing_pdfs.keys():
            return pdfname
        else:
            raise IOError("%s must specify a valid PDF name" % pdfname)

    def split_DFE_params(self, pdf_params, pdfname):
        '''
        Split PDF upper bound, lower bound in grid search
        '''
        pdf_params = list(map(float, pdf_params.split(",")))

        # check that the demog_parameters has the same length
        param_names = self.existing_pdfs[pdfname]
        if len(pdf_params) == len(param_names):
            pdf_paramdict = list(zip(param_names, pdf_params, strict=True))
        else:
            raise IOError('Mismatching pdf_params and pdfname')

        return pdf_params, pdf_paramdict

    def get_DFE_pdf(self, pdfname):
        pdf = self.pdf_function[pdfname]
        optimizer = self.Inference_optimizer[pdfname]
        integrate_methods = self.Spectra_integrate_methods[pdfname]
        return pdf, optimizer, integrate_methods


