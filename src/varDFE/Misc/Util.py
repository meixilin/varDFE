
import os
import dadi
from varDFE.Misc import LoggerDFE
from scipy.stats import variation

def ExistingFile(fname):
    """
    Return *fname* if existing file, otherwise raise IOError.
    """
    if os.path.isfile(fname):
        return fname
    else:
        raise IOError("%s must specify a valid file name" % fname)

def CreateNewDir(dirname, warning=False):
    """
    Check if directory exists. If not create one recursively.
    """
    if not os.path.isdir(dirname):
        if warning:
            LoggerDFE.logWARN("Making output directory {0}".format(dirname))
        os.makedirs(dirname)
    return None

def GetFuncName(func):
    return str(func.__module__) + "." + str(func.__name__)

def LoadFoldSFS(sfs,mask1=False):
    '''
    Load SFS and check if it is folded.
    mask1: whether or not mask singletons. default: False
    '''
    sfs=ExistingFile(sfs)
    fs=dadi.Spectrum.from_file(sfs) # this is folded if from easy SFS

    # check if it's folded, if not folded, fold it
    if fs.folded==False:
        LoggerDFE.logWARN('{0} not folded.'.format(sfs))
        fs=fs.fold()
    else:
        pass

    # check if need to mask singletons
    if mask1==True:
        LoggerDFE.logINFO('Masked singleton from input {0}'.format(sfs))
        fs.mask[1] = True

    return fs

def CheckConvergence(dt,params,topn=20):
    """
    Check for convergence across runs. Use CV as metrics for parameters. NOT testing for convergence, just output the values.
    previously: CV < 0.8 and LL1-LL20 < 100
    """
    # if dt has less than 20 rows
    topn = min(dt.shape[0],topn)
    newdt = dt[:topn]
    newdt = newdt.reset_index(drop=True)
    outconvergence = {}
    lldiff = newdt.ll_model[0] - newdt.ll_model[topn-1]
    # if lldiff > 100:
    #     outconvergence['ll_model'] = lldiff
    # else:
    #     outconvergence['ll_model'] = True
    outconvergence['ll_model'] = lldiff
    for ii in params:
        CVpar = variation(newdt[ii])
        # if CVpar > 0.8:
        #     outconvergence[ii] = CVpar
        # else:
        #     outconvergence[ii] = True
        outconvergence[ii] = CVpar
    return outconvergence

def CalcLsLns(Lcds: float, NS_S_scaling: float):
    """
    Calculate Lsyn and Lnonsyn from only Lcds and NS_S_scaling
    Lcds: length of CDS
    NSSscaling: Scaling factor = Lns/Ls
    """
    Lsyn=Lcds/(1+NS_S_scaling)
    Lsyn=Lsyn//1 # floor on Lsyn and keep float format
    Lnonsyn=Lcds-Lsyn
    return (Lsyn, Lnonsyn)

def CalcNanc(theta, mu, L):
    Nanc = theta / (4*mu*L)
    return Nanc
