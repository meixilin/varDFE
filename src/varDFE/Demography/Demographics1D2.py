'''
Additional demographics scripts
'''

from dadi import Numerics, PhiManip, Integration, Spectrum

def four_epoch(params, ns, pts):
    """
    params = (nuB,nuR,nuF,TB,TR,TF)
    ns = (n1,)

    nuB: Ratio of Bottleneck population size to ancient pop size
    nuR: Ratio of Recovery population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TR: Length of recovery (in units of 2*Na generations)
    TF: Time since contemporary size change (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuR,nuF,TB,TR,TF = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TR, nuR)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


