"""
Output demography for workflow associated with exploring demographic inference.
"""

import numpy as np
from varDFE.Demography.DemogValidation import DemogValidation
from varDFE.Misc import LoggerDFE

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def demog_scaling(Nanc, popt, modelname):
    """
    Model specific scaling of parameters by Nanc
    """
    # the first half is nua,nub,nuc; the second half is Ta,Tb,Tc
    # nu_scaled --> in units of diploid
    # T_scaled --> in units of generations
    if modelname == 'one_epoch':
        scaled_popt = None
    else:
        nepoch=int(len(popt)/2)
        scaled_popt=np.concatenate((popt[:nepoch]*Nanc,popt[nepoch:]*2*Nanc))
    return scaled_popt

def join_params(params):
    if type(params) == str or params is None:
        return params
    else:
        return ','.join(str(x) for x in params)

def write_fim_result(funcvars,params: list,theta: list,params_sd,outfile):
    with open(outfile, 'w') as outf:
        outf.write('\t'.join(funcvars)+'\ttheta\n')
        outf.write('\t'.join([str(ii) for ii in params+theta])+'\n')
        outf.write('\t'.join([str(ii) for ii in params_sd])+'\n')
    return None

def write_demog_result(args, results, popt, scaled_popt, outprefix):
    """
    Output demographic results
    results: output not included in args, popt and scaled_popt
    """
    # split results
    runNum, maxiter, ns, func, upperbound, lowerbound, p0, theta, ll_model, ll_data, Nanc = results
    modelname = args['modelname']

    # organize variables that need to be concatenated
    upper_bound = join_params(upperbound)
    lower_bound = join_params(lowerbound)
    initval = join_params(args['initval'])
    initval_p0 = join_params(p0)

    # organize outputs
    additional_data = [runNum,LoggerDFE.get_today(),maxiter,
        args['pop'],args['sfs'],args['mask_singleton'],ns[0],
        func.__name__,args['mu'],args['Lcds'],args['NS_S_scaling'],args['Lsyn'],
        upper_bound,lower_bound,initval,initval_p0,
        theta,ll_model,ll_data, Nanc]
    if modelname == 'one_epoch':
        output_data = additional_data
    else:
        output_data = additional_data + popt.tolist() + scaled_popt.tolist()

    # organize output names
    additional_data_names = ['runNum', 'rundate','maxiter',
        'pop','sfs','mask_singleton','ns',
        'demog_func', 'mu', 'Lcds', 'NS_S','Lsyn',
        'upper_bound','lower_bound','initval','initval_p0',
        'theta','ll_model','ll_data', 'Nanc']
    if modelname == 'one_epoch':
        output_names = additional_data_names
    else:
        popt_names=DemogValidation().existing_models[modelname]
        scaled_popt_names=[ii+'_sc' for ii in popt_names]
        output_names = additional_data_names + popt_names + scaled_popt_names

    # write output
    outfile = outprefix + '.txt'
    with open(outfile, 'w') as outf:
        outf.write('\t'.join(str(x) for x in output_names)+'\n')
        outf.write('\t'.join(str(x) for x in output_data)+'\n')

    # output data and names
    return output_data, output_names

# similar to the OutputDFE
def append_info(convergencedict, funcvars, fim_sd, bestprefix):
    with open(bestprefix+'.txt','rt') as infile:
        inlines = infile.readlines()
    # append headings
    newheadings=[]
    newvalues=[]
    for key, value in convergencedict.items():
        newheadings.append(key+"_converg")
        newvalues.append(value)
    newheadings=newheadings+[ii+'_sd' for ii in funcvars]+['theta_sd']
    newvalues=newvalues+fim_sd.tolist()
    outheadings="{0}\t{1}\n".format(inlines[0].strip(),'\t'.join(newheadings))
    outvalues="{0}\t{1}\n".format(inlines[1].strip(),'\t'.join([str(ii) for ii in newvalues]))
    with open(bestprefix+'.info.txt','w') as outf:
        outf.write(outheadings)
        outf.write(outvalues)
    return None
