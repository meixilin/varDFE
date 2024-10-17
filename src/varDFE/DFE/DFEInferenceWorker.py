"""
multiprocessing DFE inference worker script
"""

import dadi
from varDFE.Misc import LoggerDFE, Plotting, Util
from varDFE.DFE import OutputDFE

def DFEInferenceWorker(inputlist):
    runNum, initval, lowerbound, upperbound, pdfname, optimizer, fs, ref_spectra, pdf, args, maxiter, ns, pdfvars, integrate_methods, optimizer_name = inputlist
    runNumstr=str(runNum).zfill(2) # use this as output file name
    p0_sel = Util.perturb_params(
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

        # Calculate the best-fit model AFS.
        model=ref_spectra.integrate_continuous_pos(
            params=popt,
            ns=None,
            sel_dist=pdf,
            theta=args['theta_nonsyn'],
            pts=None)
    elif pdfname == 'shifted_gamma':
        popt = optimizer(
            p0=p0_sel,
            data=fs,
            model_func=ref_spectra.integrate_continuous_pos,
            pts=None,
            func_args=[pdf, args['theta_nonsyn']],
            lower_bound=lowerbound,
            upper_bound=upperbound,
            verbose=5,
            maxiter=maxiter,
            multinom=False)

        # Calculate the best-fit model AFS.
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

        # Calculate the best-fit model AFS.
        model=ref_spectra.integrate(
            params=popt,
            ns=None,
            sel_dist=pdf,
            theta=args['theta_nonsyn'],
            pts=None)

    # Poisson Likelihood of the data given the model AFS.
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

    ##### Output plot (same for anymodel)
    outputFigure = Plotting.plot_dadi_1d(outprefix,model,fs)

    ##### Output SFS (same for anymodel)
    model.pop_ids = [fs.pop_ids[0]+'.'+pdfname+runNumstr]
    model.to_file(outprefix + '_unfolded.expSFS')
    model_fold = model.fold()
    model_fold.to_file(outprefix + '_folded.expSFS')

    LoggerDFE.logINFO('Rep{0}. Output *_unfolded.expSFS, *_folded.expSFS, *.png, *.txt to {1}'.format(runNumstr, outprefix))
    return output_data, output_names
