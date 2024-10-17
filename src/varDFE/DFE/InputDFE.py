"""
Input validations for workflow associated with exploring DFE inference.
"""

from varDFE.Misc import LoggerDFE, Util
from varDFE.DFE.PDFValidation import PDFValidation
from varDFE.Demography.DemogValidation import DemogValidation
import argparse
import os

def parse_GridSearchArgs():
    '''
    Parse command-line arguments for workflow: DFE1D_gridsearch.py
    '''
    parser = argparse.ArgumentParser("Given the reference spectra, a range of functional form parameters (e.g. gamma shape and scale) to search from, this script outputs the likelihood surface.")

    parser.add_argument(
        "--max_bound",type=str,required=True,
        help="parameters maximum values range. provide in the order it defined. separate by comma. If scaling is needed, provide values before scaling (ie. `s` but not `NeS`) and specify dfe_scaling option.")

    parser.add_argument(
        "--min_bound",type=str,required=True,
        help="parameters minimum values range. provide in the order it defined. separate by comma.")

    parser.add_argument(
        "--dfe_scaling",action='store_true',default=False,
        help="Use `2NeS` or `S` distribution. default=False -- Use `2NeS`")

    parser.add_argument(
        "--Npts",type=int,required=False,default=250,
        help="Number of grid points per parameter you want. Keep in mind this can drastically affect run time (e.g. 10 --> 100 calculations). Default: 250.")

    parser.add_argument(
        "--Nanc",type=float,required=False,
        help="If dfe_scaling is needed. Provide Nanc")

    parser.add_argument(
        "--mask_singleton",action='store_true',default=False,
        help="mask singleton in the input SFS")

    parser.add_argument(
        "sfs",type=Util.ExistingFile,
        help="path to FOLDED NONSYN SFS in dadi format from easysfs (mask optional)")

    parser.add_argument(
        "ref_spectra",type=Util.ExistingFile,
        help="path to reference DFE spectra")

    parser.add_argument(
        "pdfname", type=PDFValidation().ExistingPDF,
        help='DFE functional form to use.')

    parser.add_argument(
        'theta_nonsyn', type=float,
        help='Theta of nonsynonymous regions from demographic inference.')

    parser.add_argument(
        "outprefix", type=str,
        help="Path/NamePrefix to the output file")

    # get args
    args = vars(parser.parse_args())

    # now only gamma distribution
    if args['pdfname'] not in ['gamma']:
        raise IOError('Only gamma distribution DFE scaling suport')

    # convert the values from max and min bounds
    for ii in ['max_bound','min_bound']:
        pdf_params0 = args[ii]
        if '"' in pdf_params0:
            pdf_params0=pdf_params0.strip('"')
        pdf_params, pdf_paramdict = PDFValidation().split_DFE_params(pdf_params=pdf_params0,pdfname=args['pdfname'])
        LoggerDFE.logINFO('DFE {0} : {1}'.format(ii,LoggerDFE.join_zip(pdf_paramdict, sep = ',')))
        args[ii] = pdf_params

    # check if Nanc is available if needs unscaling
    if args['dfe_scaling'] is True:
        if args['Nanc'] is None:
            raise IOError('Nanc is required for dfe_scaling. Provide Nanc through --Nanc')
    else:
        pass

    # check if directory exists
    outdir = os.path.dirname(args['outprefix'])
    Util.CreateNewDir(outdir)

    # log the input statistics
    LoggerDFE.print_IO(args)

    return args

def parse_InferenceArgs():
    '''
    Parse command-line arguments for workflow: DFE1D_inferenceFIM.py
    '''
    parser = argparse.ArgumentParser(description="Run DFE inference from precomputed spectra for each species/population.")

    parser.add_argument(
        "--pop",type=str,required=True,
        help="population identifier, e.g. 'HS100'")

    parser.add_argument(
        "--mu",type=float,required=True,
        help="supply exon mutation rate in mutation/bp/gen")

    parser.add_argument(
        "--Lcds",type=float,required=True,
        help="number of called CDS sites that went into making SFS (monomorphic+polymorphic)")

    parser.add_argument(
        "--NS_S_scaling",type=float,required=True,
        help="The scaling factor in Lsyn and Lnonsyn length. Lsyn + Lnonsyn(=NS_S_scaling*Lsyn) = Lcds")

    parser.add_argument(
        "--Nrun",type=int,required=False,default=100,
        help="Number of iterations e.g. 100 runs")

    parser.add_argument(
        "--mask_singleton",action='store_true',default=False,
        help="mask singleton in the input SFS")

    parser.add_argument(
        "sfs",type=Util.ExistingFile,
        help="path to FOLDED NONSYN SFS in dadi format from easysfs (mask optional)")

    parser.add_argument(
        "ref_spectra",type=Util.ExistingFile,
        help="path to reference DFE spectra")

    parser.add_argument(
        "pdfname", type=PDFValidation().ExistingPDF,
        help='DFE functional form to use.')

    parser.add_argument(
        'theta_syn', type=float,
        help='Theta of synonymous regions from demographic inference.')

    parser.add_argument(
        "outdir",type=str,
        help="path to output directory")

    # get args
    args = vars(parser.parse_args())

    # create output directory including the detailed directories
    Util.CreateNewDir(dirname = args['outdir'], warning=True)
    if args['outdir'][-1] == '/':
        args['outdir'] = args['outdir'][:-1] # trim '/' if exists in input
    Util.CreateNewDir(dirname = args['outdir']+'/detail_'+str(args['Nrun'])+'runs')

    # calculate Lnonsyn (floored):
    args['Lsyn'], args['Lnonsyn'] = Util.CalcLsLns(Lcds=args['Lcds'], NS_S_scaling =args['NS_S_scaling'])

    # calculate theta_nonsyn
    args['theta_nonsyn'] = args['theta_syn']*args['NS_S_scaling']
    # calculate Nanc
    args['Nanc'] = Util.CalcNanc(theta=args['theta_syn'],mu=args['mu'],L=args['Lsyn'])

    # log the input statistics
    LoggerDFE.print_IO(args)

    return args

def parse_SpectraArgs():
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser(description="Generate a demographics informed precomputed spectra for each species/population.")

    parser.add_argument(
        'demog_model', type=DemogValidation().ExistingModel,
        help='Demographic model to use.')

    parser.add_argument(
        'demog_params', type=str,
        help='Demographic parameters for demog+sel models. Please provide in the forms of `nu`,`T`. must be "," delimited. Make sure the values are in the correct orders (e.g."1.89,0.29" for 2Epoch or "1.8,1.2,0.32,0.28" for 3Epoch')

    parser.add_argument(
        'ns', type=int,
        help='Number of samples in the SFS to generate')

    parser.add_argument(
        "outprefix", type=str,
        help="Path/NamePrefix to the output file")

    # get args
    args = vars(parser.parse_args())
    # log the output
    LoggerDFE.print_IO(args)

    # convert the values for demography
    demog_params0=args['demog_params']
    #strip quotes from initval if inputted
    if '"' in demog_params0:
        demog_params0=demog_params0.strip('"')
    demog_params, demog_paramdict = DemogValidation().split_demog_params(
        demog_params=demog_params0, demog_model = args['demog_model'])
    args['demog_params'] = demog_params
    LoggerDFE.logINFO('Demographic params {0}'.format(LoggerDFE.join_zip(demog_paramdict, sep = ',')))

    # check if directory exists
    outdir = os.path.dirname(args['outprefix'])
    Util.CreateNewDir(outdir)

    return args


