# -*- coding: utf-8 -*-
'''
Title: variations in DFE across species workflow step0.
Generate a demographics informed precomputed spectra for each species/population.
Author: Meixi Lin
Date: 2022-04-10 11:19:27
Example usage:
python3 DFE1D_refspectra.py [-h] demog_model demog_params ns outprefix
'''

################################################################################
## import packages
import sys
import os
import dadi
import pickle

from varDFE.DFE import Cache1D_mod2, InputDFE
from varDFE.DFE.Cache1D_util import positive_gammas, dict_spectra
from varDFE.Demography.DemogValidation import DemogValidation
from varDFE.Misc import LoggerDFE, Plotting

################################################################################
## main
def main():
    # parse arguments
    args = InputDFE.parse_SpectraArgs()

    # prepare for output file
    outfile = '{0}_DFESpectrum.bpkl'.format(args['outprefix'])
    if os.path.isfile(outfile):
        LoggerDFE.logWARN("Removing {0}".format(outfile))
        os.remove(outfile)
    # find the demographic model with selection
    # note that in Cache1D it creates a extrap function
    func = DemogValidation().get_DFE_func_ex(demog_model = args['demog_model'])
    LoggerDFE.logINFO('Beginning reference spectra using DFE_demog_function {0}.'.format(func))
    # generate spectra (negative spectra + neutral + positive)
    # numbers here is to make sure the step size is 0.01 in both positive and negative spectras
    pos_gammas = positive_gammas(pos_gamma_bounds=(1e-5,100),pos_gamma_pts=701)
    ref_spectra = Cache1D_mod2.Cache1D(
        params = args['demog_params'],
        ns = [args['ns']],
        demo_sel_func = func,
        pts_l= [1000,1200,1400],
        gamma_bounds=(1e-5, 10000),
        additional_gammas=pos_gammas,
        gamma_pts=901,
        verbose=True, mp=True)

    # summary info
    LoggerDFE.logINFO('Number of negative gammas: {0}. Number of all gammas: {1}'.format(ref_spectra.neg_gammas.shape,ref_spectra.gammas.shape))

    # save spectra first
    pickle.dump(ref_spectra, open(outfile,'wb'))

    # plot the most beneficial, neutral and deleterious variations
    pp = Plotting.ggplot_ref_spectra_1d(outprefix=args['outprefix']+'_DFESpectrum_QC',
        dictspectra=dict_spectra(spectra_cache=ref_spectra))

    LoggerDFE.logEND('Spectra saved to {0}'.format(outfile))

if __name__ == "__main__":
    sys.exit(main())

