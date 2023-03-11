"""
Additional Probability density functions for defining DFEs.
"""

import numpy as np
import scipy.stats.distributions as ssd
from dadi import DFE
from mpmath import mp
mp.dps = 50

# huber et al. 2017 with
# modified mins at 1e-5 instead of 1e-4 to avoid plotting problem
# gamma distributed dfe with elevated neutral proportions
# here pneu is not a point mass, but shifts up gamma by pneu/mins
# adapted from: https://dadi.readthedocs.io/en/latest/user-guide/dfe-inference/
def neugamma(xx, params):
    """
    Define a neutral-gamma distribution.
    params: [pneu, alpha, beta] = [pneu, shape, scale]
    mins: at which gamma do we start treating s as neutral
    """
    mins = 1e-5
    pneu, alpha, beta = params
    # Convert xx to an array
    xx = np.atleast_1d(xx)
    out = (1-pneu)*DFE.PDFs.gamma(xx, (alpha, beta))
    # Assume gamma < 1e-5 is essentially neutral
    out[np.logical_and(0 <= xx, xx < mins)] += pneu/mins
    # Reduce xx back to scalar if it's possible
    return np.squeeze(out)

# lethal + gamma
def gammalet(xx, params):
    """
    Define a lethal-gamma distribution.
    params: [plet, alpha, beta] = [plet, shape, scale]
    """
    mins = 1e-5
    maxs = 10001
    plet, alpha, beta = params
    # Convert xx to an array
    xx = np.atleast_1d(xx)
    out = (1-plet)*DFE.PDFs.gamma(xx, (alpha, beta))
    # # add the lethal mass at 10001-10002 (ie. larger than the max refspectra)
    # out[np.logical_and(maxs <= xx, xx < maxs+1)] += plet
    # Reduce xx back to scalar if it's possible
    return  np.squeeze(out)


# lethal + neugamma
# adapted from: https://github.com/emmaewade/Lethals_Project/blob/main/inference.py
# in this PDF, the PDF was not expected to integrate to one. Instead it integrate to 1-plethal
def neugammalet(xx, params):
    """
    Define a lethal-neutral-gamma distribution.
    params: [pneu, plet, alpha, beta] = [pneu, plet, shape, scale]
    """
    mins = 1e-5
    maxs = 10001
    plet, pneu, alpha, beta = params
    # Convert xx to an array
    xx = np.atleast_1d(xx)
    out = (1-pneu-plet)*DFE.PDFs.gamma(xx, (alpha, beta))
    # Assume gamma < 1e-5 (at dadi website: 1e-4) is essentially neutral
    out[np.logical_and(0 <= xx, xx < mins)] += pneu/mins
    # # add the lethal mass at 10001-10002 (ie. larger than the max refspectra)
    # out[np.logical_and(maxs <= xx, xx < maxs+1)] += plet
    # Reduce xx back to scalar if it's possible
    return  np.squeeze(out)

# lourenco equilibrium eq. 15
# adapted from Huber et al. 2017 implementation
# TODO: not all values integrate to one
def lourenco_eq(xx, params):
    """
    Define a FGM based mutation-selection-drift balance DFE
    params: [m, sigma, Ne, Ne_dadi] = [pleiotropy, variation, Ne, Ne_dadi]
    """
    m, sigma, Ne, Ne_dadi = params # Ne_dadi is not estimated
    s = xx/(2.0*Ne_dadi)
    prob = (mp.power(2, (1-m)/2.)*mp.power(Ne, 0.5)*mp.power(mp.fabs(s), (m-1)/2.)*(1+1/(Ne*mp.power(sigma, 2.)))*mp.exp(-Ne*s) / (mp.power(mp.pi,0.5)*mp.power(sigma, m)*mp.gamma(m/2.))) * mp.besselk((m-1)/2., Ne*mp.fabs(s)*mp.power(1+1/(Ne*mp.power(sigma, 2.)),0.5))
    # the probability is scaled for s, convert to xx
    out = float(prob/(2.0*Ne_dadi))
    return out

# wrap around lourenco_eq
def lourenco_eq_pdf(xx, params):
    if isinstance(xx, int) or isinstance(xx, float):
        return lourenco_eq(xx, params)
    else:
        out = [lourenco_eq(x, params) for x in xx]
        return np.array(out)
