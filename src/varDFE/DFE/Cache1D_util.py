"""
Additional Cache1D module utility scripts to include positive gammas systematically like the negative gammas.
Keeping separate for clarity purposes.
"""

import numpy as np

def positive_gammas(pos_gamma_bounds, pos_gamma_pts):
    """
    Define values to input to `additional_gammas` for Cache1D
    """
     #Create a vector of positive gammas that are log-spaced over an interval
    pos_gammas = np.logspace(np.log10(pos_gamma_bounds[0]),
                             np.log10(pos_gamma_bounds[1]),
                             pos_gamma_pts)
    return pos_gammas

def dict_spectra(spectra_cache):
    dictspectra=dict(zip(spectra_cache.gammas, spectra_cache.spectra))
    dictspectra[0]=spectra_cache.neu_spec
    return dictspectra


