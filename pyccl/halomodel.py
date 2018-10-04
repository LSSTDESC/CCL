from . import ccllib as lib
from .pyutils import _vectorize_fn2, _vectorize_fn4


def halo_concentration(cosmo, halo_mass, a, odelta=200):
    """Halo mass concentration relation

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.
        odelta (float): overdensity parameter (default: 200)

    Returns:
        float or array_like: Dimensionless halo concentration, ratio rv/rs
    """
    return _vectorize_fn4(
        lib.halo_concentration,
        lib.halo_concentration_vec, cosmo, halo_mass, a, odelta)


def onehalo_matter_power(cosmo, k, a):
    """One-halo term for matter power spectrum assuming NFW density profiles
    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        onehalo_matter_power (float or array_like): one-halo term for matter
                                                    power spectrum
    """
    return _vectorize_fn2(lib.onehalo_matter_power,
                          lib.onehalo_matter_power_vec,
                          cosmo, k, a)


def twohalo_matter_power(cosmo, k, a):
    """Two-halo term for matter power spectrum assuming NFW density profiles
    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        two-halo matter power spectrum (float or array_Like): two-halo term
                                                              for matter power
                                                              spectrum
    """
    return _vectorize_fn2(
        lib.twohalo_matter_power,
        lib.twohalo_matter_power_vec,
        cosmo, k, a)


def halomodel_matter_power(cosmo, k, a):
    """Matter power spectrum from halo model assuming NFW density profiles
    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        halomodel_matter_power (float or array_like): matter power spectrum
                                                      from halo model
    """
    return _vectorize_fn2(
        lib.halomodel_matter_power,
        lib.halomodel_matter_power_vec,
        cosmo, k, a)
