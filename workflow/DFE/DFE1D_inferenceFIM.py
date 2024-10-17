# -*- coding: utf-8 -*-
'''
Title: Infer DFE for each species using the precomputed spectra
Author: Meixi Lin
Date: 2022-04-22 11:26:01
Example usage:
python3 DFE1D_inferenceFIM.py [-h] [--Nrun 20] --pop 'HS100' --mu '2.5e-8' --Lcds '19089129'
    --NS_S_scaling NS_S_SCALING '2.31' [--mask_singleton]
    sfs ref_spectra pdfname theta_syn outdir
Before commit id: fe712c33dc57c9d3f0be82ff32df8680ff2bc256. The DFE1D_inferenceFIM was single threaded. Now this workflow is multiprocess by default.
'''

################################################################################
## import packages
import sys
import os
import dadi
import pickle
import multiprocessing
import pandas as pd
import numpy as np

from varDFE.DFE.PDFValidation import PDFValidation
from varDFE.Misc import LoggerDFE, Plotting, Util
from varDFE.DFE import InputDFE, OutputDFE
from varDFE.DFE.DFEInferenceWorker import DFEInferenceWorker

################################################################################
## def variables
maxiter=100
cputouse = min(multiprocessing.cpu_count()-1, 20)

################################################################################
## main
def main():
    # parse arguments
    args = InputDFE.parse_InferenceArgs()
    pdfname=args['pdfname'] # name of the distribution model

    ##### Input data
    fs=Util.LoadFoldSFS(sfs=args['sfs'],mask1=args['mask_singleton'])
    ns=fs.sample_sizes
    ref_spectra=pickle.load(open(args['ref_spectra'],'rb'))

    ##### Set up Specific Model
    pdf, optimizer, integrate_methods =PDFValidation().get_DFE_pdf(pdfname=pdfname)
    optimizer_name = Util.GetFuncName(optimizer)
    pdfvars=PDFValidation().existing_pdfs[pdfname]
    upperbound, lowerbound, initval = PDFValidation().query_params(pdfname=pdfname)
    LoggerDFE.logINFO('Beginning DFE optimization {0} assuming PDF {1}. Total runs = {2}.\n\tparams={3}\n\tupper_bound = {4}\n\tlower_bound = {5}\n\tinitial_val = {6}'.format(
        optimizer_name, pdf, args['Nrun'], pdfvars,upperbound,lowerbound,initval))

    ##### Carry out optimization
    # setup worker input list
    listofinputs=[]
    for runNum in range(args['Nrun']):
        DFEinputs = [runNum, initval, lowerbound, upperbound, pdfname, optimizer, fs, ref_spectra, pdf, args, maxiter, ns, pdfvars, integrate_methods, optimizer_name]
        listofinputs.append(DFEinputs)
    # run optimization in cputosue processes
    with multiprocessing.Pool(processes=cputouse) as pool:
        listofresults0 = pool.map(DFEInferenceWorker, listofinputs)
    # obtain the output_data and output_names
    listofresults = [result[0] for result in listofresults0]
    output_names = listofresults0[0][1]

    ##### Sort the results and check for convergence
    sumprefix = '{0}/{1}_DFE_{2}_'.format(args['outdir'], args['pop'], pdfname)
    sumdata = pd.DataFrame(data = listofresults, columns=output_names)
    sumdata.sort_values(by = ['ll_model'], ascending=False, inplace = True)
    sumdata = sumdata.reset_index(drop=True) # remove previous indices
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    LoggerDFE.logINFO('Top 3 runs:')
    print(sumdata[:3])
    sumdata.to_csv(sumprefix+'summary.txt',sep='\t')
    # check for convergence
    LoggerDFE.logINFO('Convergence of parameters:')
    convergencedict = Util.CheckConvergence(dt=sumdata, params=pdfvars)
    print(convergencedict)

    #### Finding the best run
    # copy the top one run to a new directory
    bestrunNumstr=str(sumdata.runNum.values[0]).zfill(2)
    bestprefix0='{0}/detail_{1}runs/{2}_DFE_{3}_run{4}'.format(
        args['outdir'], args['Nrun'],args['pop'], pdfname, bestrunNumstr)
    bestrundir=args['outdir']+'/bestrun'
    Util.CreateNewDir(dirname=bestrundir)
    os.system('cp {0}* {1}/'.format(bestprefix0, bestrundir))

    # get the best run values from sumdata
    best_params = [sumdata[ii].values[0] for ii in pdfvars]

    # plot the one best run in ggplot2
    best_model_fold = Util.LoadFoldSFS(sfs=bestprefix0+'_folded.expSFS',mask1=args['mask_singleton'])
    Plotting.ggplot_dadi_1d(outprefix=sumprefix+'SFS',model=best_model_fold, fs=fs,yvar='count')
    # plot the pdf as well
    Plotting.ggplot_dfe_pdf(outprefix=sumprefix+'PDF',pdf=pdf,params=np.array(best_params))

    #### Fisher's Information Matrix (func_ex in demography).
    # Some input for lambda was not accessed (i.e. ns, grid_pts)`params, ns, grid_pts` because you need these for Godambe.py to get correct number of inputs.
    integrate_func = lambda params, ns, grid_pts: ref_spectra.integrate(params=params, ns=None, sel_dist=pdf, theta=args['theta_nonsyn'])

    # get standard deviation of the best parameter values
    # sometimes you get nan --> model not optimized
    # https://groups.google.com/g/dadi-user/c/IvSRXjmAcwc/m/WOy6tTDlBgAJ
    fim_sd = dadi.Godambe.FIM_uncert(
        func_ex=integrate_func,
        grid_pts=[],
        p0=np.array(best_params),
        data=fs,
        multinom=False)

    # output to best_run folder
    bestprefix='{0}/bestrun/{1}_DFE_{2}_run{3}'.format(
        args['outdir'],args['pop'], pdfname, bestrunNumstr)
    OutputDFE.write_fim_result(pdfvars,best_params,fim_sd,bestprefix+'.SD.txt')

    #### append the stddev and convergence info to the best run
    OutputDFE.append_info(convergencedict, pdfvars, fim_sd, bestprefix)
    LoggerDFE.logINFO('Best params STDEV = {0}'.format(fim_sd))

    LoggerDFE.logEND('DFE inference')


if __name__ == "__main__":
    sys.exit(main())

