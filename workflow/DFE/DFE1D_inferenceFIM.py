# -*- coding: utf-8 -*-
'''
Title: Infer DFE for each species using the precomputed spectra
Author: Meixi Lin
Date: 2022-04-22 11:26:01
Example usage:
python3 DFE1D_inferenceFIM.py [-h] [--Nrun 20] --pop 'HS100' --mu '2.5e-8' --Lcds '19089129'
    --NS_S_scaling NS_S_SCALING '2.31' [--mask_singleton]
    sfs ref_spectra pdfname theta_syn outdir
'''

################################################################################
## import packages
import sys
import os
import dadi
import pickle
import pandas as pd
import numpy as np

from varDFE.DFE.PDFValidation import PDFValidation
from varDFE.Misc import LoggerDFE, Plotting, Util
from varDFE.DFE import InputDFE, OutputDFE

################################################################################
## def variables
maxiter=100

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
    sumdict = {} # summary dictionary
    for runNum in range(args['Nrun']):
        runNumstr=str(runNum).zfill(2) # use this as output file name
        p0_sel = dadi.Misc.perturb_params(
            params=initval,
            lower_bound=lowerbound,
            upper_bound=upperbound,
            fold=1)
        # multinom=False --> use poisson likelihoood ie. not recalculate theta
        # use optimize function for lognormal distribution. optimize_log for the rest
        # change integrate methods for lourenco distribution
        if pdfname == 'lourenco_eq':
            popt = optimizer(
                p0=p0_sel,
                data=fs,
                model_func=ref_spectra.integrate_continuous_pos,
                pts=None,
                func_args=[pdf, args['theta_nonsyn']],
                lower_bound=lowerbound,
                upper_bound=upperbound,
                fixed_params=[None, None, None, args['Nanc']],
                verbose=5,
                maxiter=maxiter,
                multinom=False)

            # Calculate the best-fit model AFS. BK: No normalization in usual cases.
            model=ref_spectra.integrate_continuous_pos(
                params=popt,
                ns=None,
                sel_dist=pdf,
                theta=args['theta_nonsyn'],
                pts=None)
        else:
            popt = optimizer(
                p0=p0_sel,
                data=fs,
                model_func=ref_spectra.integrate,
                pts=None,
                func_args=[pdf, args['theta_nonsyn']],
                lower_bound=lowerbound,
                upper_bound=upperbound,
                verbose=5,
                maxiter=maxiter,
                multinom=False)

            # Calculate the best-fit model AFS. BK: No normalization in usual cases.
            model=ref_spectra.integrate(
                params=popt,
                ns=None,
                sel_dist=pdf,
                theta=args['theta_nonsyn'],
                pts=None)

        # Poisson Likelihood of the data given the model AFS.
        # ML: not folding model AFS gives the same result as folded AFS.
        ll_model = dadi.Inference.ll(model, fs)

        # also get the LL of the data to itself (best possible ll)
        ll_data=dadi.Inference.ll(fs, fs)

        # (un)scale parameters by Na (from NeS to S) if needed
        unscaled_popt = OutputDFE.dfe_unscaling(Nanc=args['Nanc'],popt=popt, pdfname=pdfname)

        ##### Write output file
        outprefix = '{0}/detail_{1}runs/{2}_DFE_{3}_run{4}'.format(
            args['outdir'], args['Nrun'],args['pop'], pdfname, runNumstr)
        # all the results and settings
        output_data, output_names = OutputDFE.write_dfe_result(
            args,(runNumstr, maxiter, ns, pdf, pdfvars, integrate_methods, optimizer_name, upperbound, lowerbound, initval, p0_sel, ll_model, ll_data),
            popt,unscaled_popt,outprefix)
        sumdict[runNum] = output_data

        ##### Output plot (same for anymodel)
        outputFigure = Plotting.plot_dadi_1d(outprefix,model,fs)

        ##### Output SFS (same for anymodel)
        model.pop_ids = [fs.pop_ids[0]+'.'+pdfname+runNumstr]
        model.to_file(outprefix + '_unfolded.expSFS')
        model_fold = model.fold()
        model_fold.to_file(outprefix + '_folded.expSFS')

        LoggerDFE.logINFO('Rep{0}. Output *_unfolded.expSFS, *_folded.expSFS, *.png, *.txt to {1}'.format(runNumstr, outprefix))

    ##### Sort the results and check for convergence
    sumprefix = '{0}/{1}_DFE_{2}_'.format(args['outdir'], args['pop'], pdfname)
    sumdata = pd.DataFrame.from_dict(data = sumdict, orient = 'index',columns=output_names)
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

    #### Fisher's Information Matrix (func_ex in demography)
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

