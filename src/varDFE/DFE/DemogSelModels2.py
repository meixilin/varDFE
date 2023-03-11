"""
Additional Input models of demography + selection.
"""
from dadi import Numerics, Integration, PhiManip, Spectrum

def three_epoch(params, ns, pts):
    """Define a three-epoch demography with selection included.

    This method incorporates a gamma parameter.

    params = (nua, nub, Ta, Tb,gamma)
        nua: ratio of population size between epoch 1 and 2.
        nub: ratio of population size between epoch 2 and 3.
        Ta: Bottleneck/size change length between epoch 1 and 2, in units of 2 * N_a.
        Tb: Bottleneck/size change length between epoch 2 and 3, in units of 2 * N_a.
        gamma: the input for phi_1D. scaled selection coefficient, equal to 2*Nref * s, where s is the selective advantage.
    ns = (n1, )
        n1: Number of samples in resulting Spectrum object.
    pts: Number of grid points to use in integration.
    """
    nua, nub, Ta, Tb, gamma = params  # Define given parameters.

    xx = Numerics.default_grid(pts)  # Define likelihood surface.
    phi = PhiManip.phi_1D(xx)  # Define initial phi.

    # Integrate epochs.
    phi = Integration.one_pop(phi, xx, Ta, nua, gamma=gamma)  # Integrate 1 to 2.
    phi = Integration.one_pop(phi, xx, Tb, nub, gamma=gamma)  # Integrate 2 to 3.

    # Construct spectrum object.
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def four_epoch(params, ns, pts):
    """Define a four-epoch demography with selection included.

    This method incorporates a gamma parameter.

    params = (nua, nub, nuc, Ta, Tb, Tc, gamma)
        nua: ratio of population size between epoch 1 and 2.
        nub: ratio of population size between epoch 2 and 3.
        nuc: ratio of population size between epoch 3 and 4.
        Ta: Bottleneck/size change length between epoch 1 and 2, in units of 2 * N_a.
        Tb: Bottleneck/size change length between epoch 2 and 3,
            in units of 2 * N_a.
        Tc: Bottleneck/size change length between epoch 3 and 4,
            in units of 2 * N_a.
        gamma: the input for phi_1D. scaled selection coefficient, equal to 2*Nref * s, where s is the selective advantage.
    ns = (n1, )
        n1: Number of samples in resulting Spectrum object.
    pts: Number of grid points to use in integration.
    """
    nua, nub, nuc, Ta, Tb, Tc, gamma = params  # Define given parameters.

    xx = Numerics.default_grid(pts)  # Define likelihood surface.
    phi = PhiManip.phi_1D(xx)  # Define initial phi.

    # Integrate epochs.
    phi = Integration.one_pop(phi, xx, Ta, nua, gamma=gamma)  # Integrate 1 to 2.
    phi = Integration.one_pop(phi, xx, Tb, nub, gamma=gamma)  # Integrate 2 to 3.
    phi = Integration.one_pop(phi, xx, Tc, nuc, gamma=gamma)  # Integrate 3 to 4.

    # Construct spectrum object.
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

