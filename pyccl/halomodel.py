from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn3, _vectorize_fn4

def onehalo_matter_power(cosmo, a, k):
    """one-halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        onehalo_matter_power (float or array_like): one-halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.onehalo_matter_power, 
                          lib.onehalo_matter_power_vec, cosmo, a, k)

def twohalo_matter_power(cosmo, a, k):
    """two-halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        two-halo matter power spectrum (float or array_Like): two-halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.twohalo_matter_power,
			  lib.twohalo_matter_power_vec, cosmo, a, k)

def halomodel_matter_power(cosmo, a, k):
    """matter power spectrum from halo model
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        halomodel_matter_power (float or array_like): matter power spectrum from halo model
    """
    return _vectorize_fn2(lib.halomodel_matter_power,
			  lib.halomodel_matter_power_vec, cosmo, a, k)

def halo_concentration(cosmo, a, halo_mass):
    """halo concentration
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor
        halo_mass (float or array_like): mass of halo in Msun
    
    Returns:
        halo_concentration: measure of halo concentration
    """
    return _vectorize_fn2(lib.halo_concentration,
			  lib.halo_concentration_vec, cosmo, a, halo_mass)

