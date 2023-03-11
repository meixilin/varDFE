"""
Output DFE for workflow associated with exploring DFE inference.
"""

import numpy as np
import pandas as pd
from varDFE.DFE.PDFValidation import PDFValidation
from varDFE.Misc import LoggerDFE
import math

def dfe_unscaling(Nanc, popt, pdfname):
    """
    Model specific scaling of parameters by Nanc for DFE
    Nanc: ancestral population size

    """
    if pdfname == 'gamma':
        unscaled_popt=np.divide(popt, np.array([1,2*Nanc]))
    elif pdfname == 'neugamma':
        unscaled_popt=np.divide(popt, np.array([1,1,2*Nanc]))
    elif pdfname == 'neugammalet':
        unscaled_popt=np.divide(popt, np.array([1,1,1,2*Nanc]))
    elif pdfname == 'lognormal':
        # the inferred mu is: exp(mu)=2*Ne*s -->
        # unscaled mu = mu - ln(2*Ne)
        ln2Nanc=math.log(2*Nanc)
        unscaled_popt=np.subtract(popt, np.array([ln2Nanc,0]))
    else:
        unscaled_popt=popt
    return unscaled_popt

def join_params(params):
    if type(params) == str or params is None:
        return params
    else:
        return ','.join(str(x) for x in params)

def write_dfe_result(args, results, popt, unscaled_popt, outprefix):
    """
    Output DFE inference results
    results: output not included in args, popt and unscaled_popt
    """
    # split results
    runNum, maxiter, ns, pdf, pdfvars, integrate_methods, optimizer_name, upperbound, lowerbound, initval, p0_sel, ll_model, ll_data = results

    # organize variables that need to be concatenated
    upper_bound = join_params(upperbound)
    lower_bound = join_params(lowerbound)
    initval = join_params(initval)
    initval_p0 = join_params(p0_sel)

    # organize outputs
    additional_data = [runNum,LoggerDFE.get_today(),args['Nrun'],maxiter,
        args['pop'],args['sfs'],args['ref_spectra'],args['mask_singleton'],ns[0],
        pdf.__name__,integrate_methods, optimizer_name,args['mu'],args['Lcds'],args['NS_S_scaling'],args['Lnonsyn'],
        upper_bound,lower_bound,initval,initval_p0,
        args['theta_nonsyn'],ll_model,ll_data, args['Nanc']]
    output_data = additional_data + popt.tolist() + unscaled_popt.tolist()

    # organize output names
    additional_data_names = ['runNum', 'rundate','Nrun','maxiter',
        'pop','sfs','ref_spectra','mask_singleton','ns',
        'pdf_func','integrate_func','optimize_func','mu','Lcds', 'NS_S','Lnonsyn',
        'upper_bound','lower_bound','initval','initval_p0',
        'theta_nonsyn','ll_model','ll_data', 'Nanc']
    popt_names=pdfvars
    unscaled_popt_names=[ii+'_us' for ii in popt_names] # us = unscaled
    output_names = additional_data_names + popt_names + unscaled_popt_names

    # write output
    outfile = outprefix + '.txt'
    with open(outfile, 'w') as outf:
        outf.write('\t'.join(str(x) for x in output_names)+'\n')
        outf.write('\t'.join(str(x) for x in output_data)+'\n')

    # output data and names
    return output_data, output_names

def write_fim_result(pdfvars,params,params_sd,outfile):
    with open(outfile, 'w') as outf:
        outf.write('\t'.join(pdfvars)+'\n')
        outf.write('\t'.join([str(ii) for ii in params])+'\n')
        outf.write('\t'.join([str(ii) for ii in params_sd])+'\n')
    return None

def append_info(convergencedict, pdfvars, fim_sd, bestprefix):
    with open(bestprefix+'.txt','rt') as infile:
        inlines = infile.readlines()
    # append headings
    newheadings=[]
    newvalues=[]
    for key, value in convergencedict.items():
        newheadings.append(key+"_converg")
        newvalues.append(value)
    newheadings=newheadings+[ii+'_sd' for ii in pdfvars]
    newvalues=newvalues+fim_sd.tolist()
    outheadings="{0}\t{1}\n".format(inlines[0].strip(),'\t'.join(newheadings))
    outvalues="{0}\t{1}\n".format(inlines[1].strip(),'\t'.join([str(ii) for ii in newvalues]))
    with open(bestprefix+'.info.txt','w') as outf:
        outf.write(outheadings)
        outf.write(outvalues)
    return None

def args2comment(args):
    """
    Convert arguments into comment.char for output files
    """
    outlines=["#{0}\n".format(LoggerDFE.print_now())]

    for key, value in args.items():
        if key in ['max_bound','min_bound']:
            value = join_params(value) # organize variables that need to be concatenated
        line = '#{0}="{1}"\n'.format(key,value)
        outlines.append(line)
    return outlines

def write_gridsearch_result(args,pdfvars,ll_data,ll_max,ll_griddf,outfile):
    """
    Output DFE grid search results
    """
    # organize args as annotations
    outlines = args2comment(args)

    # append info
    outlines.append('#{0}="{1}"\n'.format('pdfvars',join_params(pdfvars)))
    outlines.append('#{0}="{1}"\n'.format('ll_data',ll_data))
    outlines.append('#{0}="{1}"\n'.format('ll_max',ll_max))

    with open(outfile, 'w') as outf:
        outf.writelines(outlines)
        outf.write("#ll_grid\n")

    ll_griddf.to_csv(outfile, sep='\t',mode='a')
    return None


