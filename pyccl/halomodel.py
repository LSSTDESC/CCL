"""
.. warning:: The functionality contained in this module has been deprecated
             in favour of the newer halo model implementation
             :py:mod:`~pyccl.halos`.
"""

__all__ = ("halo_concentration", "onehalo_matter_power",
           "twohalo_matter_power", "halomodel_matter_power",)

from . import deprecated, lib
from . import halos as hal


def _get_concentration(cosmo, mass_def):
    if cosmo._config.halo_concentration_method == lib.bhattacharya2011:
        c = hal.ConcentrationBhattacharya13(mdef=mass_def)
    elif cosmo._config.halo_concentration_method == lib.duffy2008:
        c = hal.ConcentrationDuffy08(mdef=mass_def)
    elif cosmo._config.halo_concentration_method == lib.constant_concentration:
        c = hal.ConcentrationConstant(c=4., mdef=mass_def)
    return c


def _get_mf_hb(cosmo, mass_def):
    if cosmo._config.mass_function_method == lib.tinker10:
        hmf = hal.MassFuncTinker10(cosmo, mass_def=mass_def,
                                   mass_def_strict=False)
        hbf = hal.HaloBiasTinker10(cosmo, mass_def=mass_def,
                                   mass_def_strict=False)
    elif cosmo._config.mass_function_method == lib.shethtormen:
        hmf = hal.MassFuncSheth99(cosmo, mass_def=mass_def,
                                  mass_def_strict=False,
                                  use_delta_c_fit=True)
        hbf = hal.HaloBiasSheth99(cosmo, mass_def=mass_def,
                                  mass_def_strict=False)
    else:
        raise ValueError("Halo model spectra not available for your "
                         "current choice of mass function with the "
                         "deprecated implementation.")
    return hmf, hbf


@deprecated(hal.Concentration)
def halo_concentration(cosmo, halo_mass, a, odelta=200):
    """Halo mass concentration relation

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        halo_mass (:obj:`float` or `array`): Halo masses; Msun.
        a (:obj:`float`): scale factor.
        odelta (:obj:`float`): overdensity parameter (default: 200)

    Returns:
        (:obj:`float` or `array`): Dimensionless halo concentration,
        ratio rv/rs.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = _get_concentration(cosmo, mdef)
    return c(cosmo, halo_mass, a)


@deprecated(hal.halomod_power_spectrum)
def onehalo_matter_power(cosmo, k, a):
    """One-halo term for matter power spectrum assuming NFW density profiles

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): wavenumber
        a (:obj:`float`): scale factor.

    Returns:
        (:obj:`float` or `array`): one-halo term for matter \
            power spectrum
    """
    mdef = hal.MassDef('vir', 'matter')
    c = _get_concentration(cosmo, mdef)
    hmf, hbf = _get_mf_hb(cosmo, mdef)
    prf = hal.HaloProfileNFW(c)
    hmc = hal.HMCalculator(cosmo, hmf, hbf, mdef)

    return hal.halomod_power_spectrum(cosmo, hmc, k, a,
                                      prf, normprof1=True,
                                      get_2h=False)


@deprecated(hal.halomod_power_spectrum)
def twohalo_matter_power(cosmo, k, a):
    """Two-halo term for matter power spectrum assuming NFW density profiles

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): wavenumber
        a (:obj:`float`): scale factor.

    Returns:
        (:obj:`float` or `array`): two-halo term for matter power spectrum.
    """
    mdef = hal.MassDef('vir', 'matter')
    c = _get_concentration(cosmo, mdef)
    hmf, hbf = _get_mf_hb(cosmo, mdef)
    prf = hal.HaloProfileNFW(c)
    hmc = hal.HMCalculator(cosmo, hmf, hbf, mdef)

    return hal.halomod_power_spectrum(cosmo, hmc, k, a,
                                      prf, normprof1=True,
                                      get_1h=False)


@deprecated(hal.halomod_power_spectrum)
def halomodel_matter_power(cosmo, k, a):
    """Matter power spectrum from halo model assuming NFW density profiles

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        k (:obj:`float` or `array`): wavenumber
        a (:obj:`float`): scale factor.

    Returns:
        (:obj:`float` or `array`): matter power spectrum from halo model
    """
    mdef = hal.MassDef('vir', 'matter')
    c = _get_concentration(cosmo, mdef)
    hmf, hbf = _get_mf_hb(cosmo, mdef)
    prf = hal.HaloProfileNFW(c)
    hmc = hal.HMCalculator(cosmo, hmf, hbf, mdef)

    return hal.halomod_power_spectrum(cosmo, hmc, k, a,
                                      prf, normprof1=True)
