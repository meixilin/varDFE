"""
multiprocessing DFE grid search worker script
"""
import dadi
import numpy as np

def DFEGridsearchWorker(inputlist):
    ref_spectra, popt, pdf, theta_nonsyn, fs = inputlist
    model=ref_spectra.integrate(
                params=popt,
                ns=None,
                sel_dist=pdf,
                theta=theta_nonsyn,
                pts=None)
    ll_model = dadi.Inference.ll(model, fs)
    output = np.append(popt, ll_model)
    return output
