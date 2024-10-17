# -*- coding: utf-8 -*-
'''
Title: Test the null model for DFE variations for each species using the precomputed spectra
No optimization was performed. The LL is calculated by adding the grids together.
This allowed flexible comparisons of models in multiple species.
Author: Meixi Lin
Date: 2023-05-10 15:23:07
Example usage:
python3 DFE1D_gridsearch.py [-h] --max_bound '0.5,2000' --min_bound '1e-5,1e-2'
    [--dfe_scaling] [--Npts 20] [--Nanc 3000] [--mask_singleton]
    ref_spectra pdfname theta_nonsyn outprefix
'''

################################################################################
## import packages
import sys
import dadi
import pickle
import multiprocessing
import pandas as pd
import numpy as np

from varDFE.DFE.PDFValidation import PDFValidation
from varDFE.Misc import LoggerDFE, Plotting, Util
from varDFE.DFE import InputDFE, OutputDFE
from varDFE.DFE.DFEGridsearchWorker import DFEGridsearchWorker

################################################################################
## def variables
cputouse = min(multiprocessing.cpu_count()-1, 20)

################################################################################
## main
def main():
    # parse arguments
    args = InputDFE.parse_GridSearchArgs()

    pdfname=args['pdfname'] # name of the distribution model
    outtxt = args['outprefix'] + '.txt'
    outnpy = args['outprefix'] + '.npy'

    ##### Input data
    fs=Util.LoadFoldSFS(sfs=args['sfs'],mask1=args['mask_singleton'])
    ref_spectra=pickle.load(open(args['ref_spectra'],'rb'))

    ##### Set up Specific Model and Parameter grids
    pdf, optimizer, integrate_methods = PDFValidation().get_DFE_pdf(pdfname=pdfname)
    pdfvars=PDFValidation().existing_pdfs[pdfname]


    # If needs to scale up to population level (reverse to dfe_unscaling)
    if args['dfe_scaling'] is True:
        # Use linear grid space
        var0_us = np.linspace(args['min_bound'][0],args['max_bound'][0],args['Npts'])
        var1_us = np.linspace(args['min_bound'][1],args['max_bound'][1],args['Npts'])
        # Currently only supports gamma
        if pdfname == 'gamma':
            var0 = var0_us.copy()
            var1 = var1_us.copy()*(2*args['Nanc'])
        else:
            raise IOError('Only gamma distribution DFE scaling suport')
    else:
        var0 = np.linspace(args['min_bound'][0],args['max_bound'][0],args['Npts'])
        var1 = np.linspace(args['min_bound'][1],args['max_bound'][1],args['Npts'])
        # Do not calculate var0_us and var1_us if testing NeS scenarios
        if pdfname == 'gamma':
            pass
            # var0_us = var0.copy()
            # var1_us = var1.copy()/(2*args['Nanc'])
        else:
            raise IOError('Only gamma distribution DFE scaling suport')

    #### Start running the grid search
    # TIPS: 3**2 is 3^2. Using all scaled values (var0, var1) as inputs.
    # setup worker input list
    listofinputs=[]
    for var0ii in var0:
        for var0jj in var1:
            popt = np.array([var0ii,var0jj])
            Gridsearchinputs = [ref_spectra, popt, pdf, args['theta_nonsyn'], fs]
            listofinputs.append(Gridsearchinputs)
    with multiprocessing.Pool(processes=cputouse) as pool:
        ll_grid0 = pool.map(DFEGridsearchWorker, listofinputs)
    ll_grid = np.array(ll_grid0)

    # also get the LL of the data to itself (best possible ll)
    ll_data=dadi.Inference.ll(fs, fs)

    # get the mle
    ll_max=ll_grid[np.argmax(ll_grid, axis=0)[2]]
    LoggerDFE.logINFO('Maximum LL in grid search ([parameters, ll_model]) = {0}'.format(ll_max))

    ##### Write output file
    # save the numpy array (didn't save var0_us and var1_us since easily reestimated)
    np.save(file = outnpy, arr = ll_grid)

    # write the text files (headers as annotations)
    ll_griddf = pd.DataFrame(data = ll_grid, columns = pdfvars+['ll_model'])

    OutputDFE.write_gridsearch_result(args,pdfvars,ll_data,ll_max,ll_griddf, outtxt)

    ##### Output plot
    pp = Plotting.ggplot_gridsearch(args['outprefix'], ll_griddf, ll_max)

    LoggerDFE.logEND('DFE grid search')




if __name__ == "__main__":
    sys.exit(main())

