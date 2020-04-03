from . import ccllib as lib
from .pyutils import deprecated
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
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.
        odelta (float): overdensity parameter (default: 200)

    Returns:
        float or array_like: Dimensionless halo concentration, ratio rv/rs
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = _get_concentration(cosmo, mdef)

    return c.get_concentration(cosmo, halo_mass, a)


@deprecated(hal.halomod_power_spectrum)
def onehalo_matter_power(cosmo, k, a):
    """One-halo term for matter power spectrum assuming NFW density profiles

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        float or array_like: one-halo term for matter \
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
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        float or array_like: two-halo term for matter power spectrum.
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
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        float or array_like: matter power spectrum from halo model
    """
    mdef = hal.MassDef('vir', 'matter')
    c = _get_concentration(cosmo, mdef)
    hmf, hbf = _get_mf_hb(cosmo, mdef)
    prf = hal.HaloProfileNFW(c)
    hmc = hal.HMCalculator(cosmo, hmf, hbf, mdef)

    return hal.halomod_power_spectrum(cosmo, hmc, k, a,
                                      prf, normprof1=True)
