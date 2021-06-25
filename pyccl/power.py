from . import ccllib as lib
import numpy as np
from .core import check
from .pk2d import parse_pk2d


def linear_power(cosmo, k, a, p_of_k_a='delta_matter:delta_matter'):
    """The linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.
        p_of_k_a (str): string specifying the power spectrum to
            compute (which should be stored in `cosmo`). Defaults to
            the linear matter power spectrum.

    Returns:
        float or array_like: Linear power spectrum.
    """
    cosmo.compute_linear_power()
    if p_of_k_a not in cosmo._pk_lin:
        raise KeyError("Power spectrum %s unknown" % p_of_k_a)
    return cosmo._pk_lin[p_of_k_a].eval(k, a, cosmo)


def nonlin_power(cosmo, k, a, p_of_k_a='delta_matter:delta_matter'):
    """The non-linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.
        p_of_k_a (str): string specifying the power spectrum to
            compute (which should be stored in `cosmo`). Defaults to
            the non-linear matter power spectrum.

    Returns:
        float or array_like: Non-linear power spectrum.
    """
    cosmo.compute_nonlin_power()
    if p_of_k_a not in cosmo._pk_nl:
        raise KeyError("Power spectrum %s unknown" % p_of_k_a)
    return cosmo._pk_nl[p_of_k_a].eval(k, a, cosmo)


def linear_matter_power(cosmo, k, a):
    """The linear matter power spectrum; Mpc^3.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Linear matter power spectrum; Mpc^3.
    """
    cosmo.compute_linear_power()
    return cosmo._pk_lin['delta_matter:delta_matter'].eval(k, a,
                                                           cosmo)


def nonlin_matter_power(cosmo, k, a):
    """The nonlinear matter power spectrum; Mpc^3.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Nonlinear matter power spectrum; Mpc^3.
    """
    cosmo.compute_nonlin_power()
    return cosmo._pk_nl['delta_matter:delta_matter'].eval(k, a,
                                                          cosmo)


def sigmaM(cosmo, M, a):
    """Root mean squared variance for the given halo mass of the linear power
    spectrum; Msun.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        M (float or array_like): Halo masses; Msun.
        a (float): scale factor.

    Returns:
        float or array_like: RMS variance of halo mass.
    """
    cosmo.compute_sigma()

    # sigma(M)
    logM = np.log10(np.atleast_1d(M))
    status = 0
    sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                len(logM), status)
    check(status)
    if np.ndim(M) == 0:
        sigM = sigM[0]
    return sigM


def sigmaR(cosmo, R, a=1., p_of_k_a=None):
    """RMS variance in a top-hat sphere of radius R in Mpc.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.
        a (float): optional scale factor; defaults to a=1
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

    Returns:
        float or array_like: RMS variance in the density field in top-hat \
            sphere; Mpc.
    """
    psp = parse_pk2d(cosmo, p_of_k_a, is_linear=True)
    status = 0
    R_use = np.atleast_1d(R)
    sR, status = lib.sigmaR_vec(cosmo.cosmo, psp, a, R_use,
                                R_use.size, status)
    check(status, cosmo)
    if np.ndim(R) == 0:
        sR = sR[0]
    return sR


def sigmaV(cosmo, R, a=1., p_of_k_a=None):
    """RMS variance in the displacement field in a top-hat sphere of radius R.
    The linear displacement field is the gradient of the linear density field.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.
        a (float): optional scale factor; defaults to a=1
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

    Returns:
        sigmaV (float or array_like): RMS variance in the displacement field \
            in top-hat sphere.
    """
    psp = parse_pk2d(cosmo, p_of_k_a, is_linear=True)
    status = 0
    R_use = np.atleast_1d(R)
    sV, status = lib.sigmaV_vec(cosmo.cosmo, psp, a, R_use,
                                R_use.size, status)
    check(status, cosmo)
    if np.ndim(R) == 0:
        sV = sV[0]
    return sV


def sigma8(cosmo, p_of_k_a=None):
    """RMS variance in a top-hat sphere of radius 8 Mpc/h.

    .. note:: 8 Mpc/h is rescaled based on the chosen value of the Hubble
              constant within `cosmo`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

    Returns:
        float: RMS variance in top-hat sphere of radius 8 Mpc/h.
    """
    return sigmaR(cosmo, 8.0 / cosmo['h'], p_of_k_a=p_of_k_a)


def kNL(cosmo, a, p_of_k_a=None):
    """Scale for the non-linear cut.

    .. note:: k_NL is calculated based on Lagrangian perturbation theory as the
              inverse of the variance of the displacement field, i.e.
              k_NL = 1/sigma_eta = [1/(6 pi^2) * int P_L(k) dk]^{-1/2}.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

    Returns:
        float or array-like: Scale of non-linear cut-off; Mpc^-1.
    """
    psp = parse_pk2d(cosmo, p_of_k_a, is_linear=True)
    status = 0
    a_use = np.atleast_1d(a)
    knl, status = lib.kNL_vec(cosmo.cosmo, psp, a_use,
                              a_use.size, status)
    check(status, cosmo)
    if np.ndim(a) == 0:
        knl = knl[0]
    return knl
