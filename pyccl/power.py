__all__ = ("linear_power", "nonlin_power", "linear_matter_power",
           "nonlin_matter_power", "sigmaM", "sigmaR", "sigmaV", "sigma8",
           "kNL",)

import numpy as np

from . import DEFAULT_POWER_SPECTRUM, check, lib


def linear_power(cosmo, k, a, *, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """The linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): Wavenumber; :math:`{\\rm Mpc}^{-1}`.
        a (:obj:`float` or `array`): Scale factor.
        p_of_k_a (:obj:`str`): string specifying the power spectrum to
            compute (which should be stored in ``cosmo``). Defaults to
            the linear matter power spectrum.

    Returns:
        (:obj:`float` or `array`): Linear power spectrum.
    """
    return cosmo.get_linear_power(p_of_k_a)(k, a, cosmo)


def nonlin_power(cosmo, k, a, *, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """The non-linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): Wavenumber; :math:`{\\rm Mpc}^{-1}`.
        a (:obj:`float` or `array`): Scale factor.
        p_of_k_a (:obj:`str`): string specifying the power spectrum to
            compute (which should be stored in ``cosmo``). Defaults to
            the non-linear matter power spectrum.

    Returns:
        (:obj:`float` or `array`): Non-linear power spectrum.
    """
    return cosmo.get_nonlin_power(p_of_k_a)(k, a, cosmo)


def linear_matter_power(cosmo, k, a):
    """The linear matter power spectrum

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): Wavenumber; :math:`{\\rm Mpc}^{-1}`.
        a (:obj:`float` or `array`): Scale factor.

    Returns:
        (:obj:`float` or `array`): Linear matter power spectrum;
        :math:`{\\rm Mpc}^3`.
    """
    return cosmo.linear_power(k, a, p_of_k_a=DEFAULT_POWER_SPECTRUM)


def nonlin_matter_power(cosmo, k, a):
    """The nonlinear matter power spectrum

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): Wavenumber; :math:`{\\rm Mpc}^{-1}`.
        a (:obj:`float` or `array`): Scale factor.

    Returns:
        (:obj:`float` or `array`): Nonlinear matter power spectrum;
        :math:`{\\rm Mpc}^3`.
    """
    return cosmo.nonlin_power(k, a, p_of_k_a=DEFAULT_POWER_SPECTRUM)


def sigmaM(cosmo, M, a):
    """RMS on the scale of a halo of mass :math:`M`. Calculated
    as :math:`\\sigma_R` (see :func:`sigmaR`) with :math:`R` being
    the Lagrangian radius of a halo of mass :math:`M` (see
    :func:`~pyccl.halos.massdef.mass2radius_lagrangian`).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        M (:obj:`float` or `array`): Halo masses.
        a (:obj:`float`): scale factor.

    Returns:
        (:obj:`float` or `array`): RMS variance of halo mass.
    """
    cosmo.compute_sigma()

    logM = np.log10(np.atleast_1d(M))
    status = 0
    sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                len(logM), status)
    check(status, cosmo=cosmo)
    if np.ndim(M) == 0:
        sigM = sigM[0]
    return sigM


def sigmaR(cosmo, R, a=1, *, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """RMS of the matter overdensity a top-hat sphere of radius :math:`R`.

    .. math::
        \\sigma_R^2(z)=\\frac{1}{2\\pi^2}\\int dk\\,k^2\\,P(k,z)\\,
        |W(kR)|^2,

    with :math:`W(x)=3(\\sin(x)-x\\cos(x))/x^3`.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        R (:obj:`float` or `array`): Radius; Mpc.
        a (:obj:`float`): optional scale factor.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`):
            power spectrum to integrate. If a string, it must correspond to
            one of the linear power spectra stored in ``cosmo`` (e.g.
            ``'delta_matter:delta_matter'``).

    Returns:
        (:obj:`float` or `array`): :math:`\\sigma_R`.
    """
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    status = 0
    R_use = np.atleast_1d(R)
    sR, status = lib.sigmaR_vec(cosmo.cosmo, psp, a, R_use, R_use.size, status)
    check(status, cosmo)
    if np.ndim(R) == 0:
        sR = sR[0]
    return sR


def sigmaV(cosmo, R, a=1, *, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """RMS of the linear displacement field in a top-hat sphere of radius R.

    .. math::
        \\sigma_V^2(z)=\\frac{1}{6\\pi^2}\\int dk\\,P(k,z)\\,|W(kR)|^2,

    with :math:`W(x)=3(\\sin(x)-x\\cos(x))/x^3`.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        R (:obj:`float` or `array`): Radius; Mpc.
        a (:obj:`float`): optional scale factor.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`):
            power spectrum to integrate. If a string, it must correspond to
            one of the linear power spectra stored in ``cosmo`` (e.g.
            ``'delta_matter:delta_matter'``).

    Returns:
        (:obj:`float` or `array`): :math:`\\sigma_V` (:math:`{\\rm Mpc}`).
    """
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    status = 0
    R_use = np.atleast_1d(R)
    sV, status = lib.sigmaV_vec(cosmo.cosmo, psp, a, R_use, R_use.size, status)
    check(status, cosmo)
    if np.ndim(R) == 0:
        sV = sV[0]
    return sV


def sigma8(cosmo, *, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """RMS variance in a top-hat sphere of radius :math:`8\\,{\\rm Mpc}/h`,
    (with the value of :math:`h` extracted from ``cosmo``) at :math:`z=0`.


    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`):
            power spectrum  to integrate. If a string, it must correspond to
            one of the linear power spectra stored in ``cosmo`` (e.g.
            ``'delta_matter:delta_matter'``).

    Returns:
        :obj:`float`: :math:`\\sigma_8`.
    """
    sig8 = cosmo.sigmaR(8/cosmo["h"], p_of_k_a=p_of_k_a)
    if np.isnan(cosmo["sigma8"]):
        cosmo._fill_params(sigma8=sig8)
    return sig8


def kNL(cosmo, a, *, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    r"""Non-linear scale :math:`k_{\rm NL}`. Calculated based on Lagrangian
    perturbation theory as the inverse of the rms of the displacement
    field, i.e.:

    .. math::
        k_{\rm NL}(z) = \left[\frac{1}{6\pi^2} \int dk\,P_L(k,z)\right]^{-1/2}.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`):
            power spectrum  to integrate. If a string, it must correspond to
            one of the linear power spectra stored in ``cosmo`` (e.g.
            ``'delta_matter:delta_matter'``).

    Returns:
        :obj:`float` or `array`: :math:`k_{\rm NL}`.
    """
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    status = 0
    a_use = np.atleast_1d(a)
    knl, status = lib.kNL_vec(cosmo.cosmo, psp, a_use, a_use.size, status)
    check(status, cosmo)
    if np.ndim(a) == 0:
        knl = knl[0]
    return knl
