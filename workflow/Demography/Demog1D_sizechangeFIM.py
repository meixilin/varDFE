# -*- coding: utf-8 -*-
'''
Title: Demographic inference for each species from SYN data. Prepare for variations in DFE across species workflow.
Author: Meixi Lin
Date: 2022-04-22 11:26:01
Example usage:
python3 Demog1D_sizechangeFIM.py [-h] [--Nrun 20] --pop 'HS100' --mu '2.5e-8' --Lcds '19089129'
    --NS_S_scaling '2.31' [--initval '0.5,0.2'] [--mask_singleton]
    sfs modelname outdir
'''

################################################################################
## import packages
import sys
import os
import dadi
import pandas as pd
import numpy as np
import func_timeout
from func_timeout import FunctionTimedOut

from varDFE.Demography.DemogValidation import DemogValidation
from varDFE.Misc import LoggerDFE, Plotting, Util
from varDFE.Demography import InputDemog, OutputDemog


################################################################################
## def variables
maxiter=100

################################################################################
## main
def main():
    # parse arguments
    args = InputDemog.parse_DemogArgs()
    modelname = args['modelname']

    ##### Input data
    fs=Util.LoadFoldSFS(sfs=args['sfs'],mask1=args['mask_singleton'])
    ns=fs.sample_sizes
    pts_l = [ns[0]+5,ns[0]+15,ns[0]+25] # slightly larger (+5) than ns and increase by 10

    ##### Set up Specific Model
    func=DemogValidation().get_Demog_func_ex(modelname=modelname)
    funcvars=DemogValidation().existing_models[modelname]
    upperbound, lowerbound = DemogValidation().query_params(modelname=modelname)
    LoggerDFE.logINFO('Beginning demography optimization using {0}. Total runs = {1}.\n\tparams={2}\n\tupper_bound = {3}\n\tlower_bound = {4}\n\tinitial_val = {5}'.format(
        func, args['Nrun'], funcvars,upperbound,lowerbound,args['initval']))

    ##### Carry out optimization
    # Make extrapolation function:
    func_ex = dadi.Numerics.make_extrap_log_func(func)
    sumdict = {} # summary dictionary
    for runNum in range(args['Nrun']):
        runNumstr=str(runNum).zfill(2) # use this as output file name
        if modelname == 'one_epoch':
            p0 = None
            popt = "N/A"
        else:
            if args['impatient'] < 0:
                # perturb parameters
                p0 = dadi.Misc.perturb_params(
                    params=args['initval'],
                    lower_bound=lowerbound,
                    upper_bound=upperbound,
                    fold=1)
                # optimize
                popt = dadi.Inference.optimize_log(
                    p0, fs, func_ex, pts_l,
                    lower_bound=lowerbound,
                    upper_bound=upperbound,
                    verbose=5,
                    maxiter=maxiter)
            else:
                # add an impatient retry parameter
                popt = []
                while len(popt) == 0:
                    try:
                        # perturb parameters
                        p0 = dadi.Misc.perturb_params(
                            params=args['initval'],
                            lower_bound=lowerbound,
                            upper_bound=upperbound,
                            fold=1)
                        # optimize, retry if timed out
                        popt = func_timeout.func_timeout(
                            timeout=args['impatient'],
                            func=dadi.Inference.optimize_log,
                            args=(p0, fs, func_ex, pts_l),
                            kwargs={'lower_bound': lowerbound,
                                    'upper_bound': upperbound,
                                    'verbose': 5,
                                    'maxiter': maxiter})
                    except FunctionTimedOut:
                        LoggerDFE.logWARN('Rep{0}. Runtime exceeded {1}s. Retrying ...'.format(runNumstr, args['impatient']))
                        popt = []
        # Calculate the best-fit model AFS.
        model = func_ex(popt, ns, pts_l)
        # Likelihood of the data given the model AFS.
        ll_model = dadi.Inference.ll_multinom(model, fs)

        # also get the LL of the data to itself (best possible ll)
        ll_data=dadi.Inference.ll_multinom(fs, fs)

        # calculate best fit theta and rescale parameters
        theta = dadi.Inference.optimal_sfs_scaling(model, fs)
        Nanc=Util.CalcNanc(theta=theta,mu=args['mu'],L=args['Lsyn'])
        scaled_popt = OutputDemog.demog_scaling(Nanc=Nanc,popt=popt,modelname=modelname)

        ##### Write output file
        outprefix = '{0}/detail_{1}runs/{2}_demog_{3}_run{4}'.format(args['outdir'], args['Nrun'],args['pop'], modelname, runNumstr)
        # all the results and settings
        output_data, output_names = OutputDemog.write_demog_result(
            args,(runNumstr, maxiter, ns, func, upperbound, lowerbound, p0, theta, ll_model, ll_data, Nanc),
            popt,scaled_popt,outprefix)
        sumdict[runNum] = output_data

        ##### Output plot (same for anymodel)
        outputFigure = Plotting.plot_dadi_1d(outprefix,model,fs)

        ##### Output SFS (same for anymodel)
        model.pop_ids = [fs.pop_ids[0]+'.'+modelname+runNumstr]
        model.to_file(outprefix + '_unfolded.expSFS')
        model_fold = model.fold()
        model_fold.to_file(outprefix + '_folded.expSFS')

        LoggerDFE.logINFO('Rep{0}. Output *_unfolded.expSFS, *_folded.expSFS, *.png, *.txt to {1}'.format(runNumstr, outprefix))

    ##### Sort the results and check for convergence
    sumprefix = '{0}/{1}_demog_{2}_'.format(args['outdir'], args['pop'], modelname)
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
    convergencedict = Util.CheckConvergence(dt=sumdata, params=funcvars)
    print(convergencedict)

    #### Finding the best run
    # copy the top one run to a new directory
    bestrunNumstr=str(sumdata.runNum.values[0]).zfill(2)
    bestprefix0='{0}/detail_{1}runs/{2}_demog_{3}_run{4}'.format(
        args['outdir'], args['Nrun'],args['pop'], modelname, bestrunNumstr)
    bestrundir=args['outdir']+'/bestrun'
    Util.CreateNewDir(dirname=bestrundir)
    os.system('cp {0}* {1}/'.format(bestprefix0, bestrundir))

    # get the best run values from sumdata
    best_params = [sumdata[ii].values[0] for ii in funcvars]
    best_theta = [sumdata['theta'].values[0]]

    # plot the one best run in ggplot2
    best_model_fold = Util.LoadFoldSFS(sfs=bestprefix0+'_folded.expSFS',mask1=args['mask_singleton'])
    Plotting.ggplot_dadi_1d(outprefix=sumprefix+'SFS',model=best_model_fold, fs=fs,yvar='percent')

    #### Fisher's Information Matrix (func_ex in demography)
    # get standard deviation of the best parameter values
    # note: there could be errors `raise LinAlgError("Singular matrix")` in not optimal conditions
    try:
        fim_sd = dadi.Godambe.FIM_uncert(
            func_ex=func_ex,
            grid_pts=pts_l,
            p0=np.array(best_params),
            data=fs,
            multinom=True)
    except np.linalg.LinAlgError:
        LoggerDFE.logWARN('Singular matrix generated during FIM computation. fim_sd not computed.')
        # fill the fim_sd to continue the pipeline
        fim_sd = np.full(shape=len(p0)+1,fill_value=np.nan)

    # output to best_run folder
    bestprefix='{0}/bestrun/{1}_demog_{2}_run{3}'.format(
        args['outdir'],args['pop'], modelname, bestrunNumstr)
    OutputDemog.write_fim_result(funcvars,best_params,best_theta,fim_sd,bestprefix+'.SD.txt')

    #### append the stddev and convergence info to the best run
    OutputDemog.append_info(convergencedict, funcvars, fim_sd, bestprefix)
    LoggerDFE.logINFO('Best params STDEV = {0}'.format(fim_sd))

    LoggerDFE.logEND('Demography inference')

if __name__ == "__main__":
    sys.exit(main())

