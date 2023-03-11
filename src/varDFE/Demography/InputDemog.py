"""
Input validations for workflow associated with exploring demographic inference.
"""

from varDFE.Misc import LoggerDFE, Util
from varDFE.Demography.DemogValidation import DemogValidation
import argparse

def parse_DemogArgs():
    '''
    Parse command-line arguments for all demography workflow scripts
    '''

    # set up parser
    parser = argparse.ArgumentParser(description='Infer a 1D one_epoch or two_epoch or three_epoch or four_epoch model from a 1D folded SFS in dadi')

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
        help="The scaling factor in Lsyn and Lnonsyn length. Lsyn + Lnonsyn(=NS_S_scaling*Lsyn)=Lcds")

    parser.add_argument(
        "--Nrun",type=int,required=False,default=100,
        help="Number of iterations e.g. 100 runs")

    parser.add_argument(
        "--initval",type=str,required=False,default=None,
        help="parameters initial starting value. provide in the order it defined. separate by comma. For 1Epoch model, no input or NA.")

    parser.add_argument(
        "--mask_singleton",action='store_true',default=False,
        help="mask singleton in the input SFS")

    parser.add_argument(
        "--impatient",type=int,default=-1,
        help="retry another p0 after ** seconds. Default to NOT retry.")

    parser.add_argument(
        "sfs",type=Util.ExistingFile,
        help="path to FOLDED SFS in dadi format from easysfs (mask optional)")
    # in demography related things, modelname is used. in DFE, this is referred to as 'demog_model'
    parser.add_argument(
        "modelname", type=DemogValidation().ExistingModel,
        help='Demographic model to use.')

    parser.add_argument(
        "outdir",type=str,
        help="path to output directory")

    args = vars(parser.parse_args())

    modelname = args['modelname']

    # convert the initial values for demography
    if modelname == 'one_epoch':
        # assign None regardless of input
        args['initval'] = None
    else:
        initval=args['initval']
        # fill in initval if there is no input
        if initval is None:
            nparams = len(DemogValidation().existing_models[modelname])
            initval = ','.join(['1']*nparams)
        #strip quotes from initval if inputted
        if '"' in initval:
            initval=initval.strip('"')
        demog_params, demog_paramdict = DemogValidation().split_demog_params(demog_params=initval, modelname = modelname)
        args['initval'] = demog_params

    # create output directory including the detailed directories
    Util.CreateNewDir(dirname = args['outdir'], warning=True)
    if args['outdir'][-1] == '/':
        args['outdir'] = args['outdir'][:-1] # trim '/' if exists in input
    Util.CreateNewDir(dirname = args['outdir']+'/detail_'+str(args['Nrun'])+'runs')

    # calculate Lsyn (floored) and Lnonsyn (though not used):
    args['Lsyn'], args['Lnonsyn'] = Util.CalcLsLns(Lcds=args['Lcds'], NS_S_scaling =args['NS_S_scaling'])

    # log the input statistics
    LoggerDFE.print_IO(args)

    return args


