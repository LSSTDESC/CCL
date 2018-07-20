from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn2, _vectorize_fn4

def onehalo_matter_power(cosmo, k, a):
    """One-halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        onehalo_matter_power (float or array_like): one-halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.onehalo_matter_power, 
                          lib.onehalo_matter_power_vec, cosmo, k, a)

def twohalo_matter_power(cosmo, k, a):
    """Two-halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        two-halo matter power spectrum (float or array_Like): two-halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.twohalo_matter_power,
			  lib.twohalo_matter_power_vec, cosmo, k, a)

def halomodel_matter_power(cosmo, k, a):
    """Matter power spectrum from halo model
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        halomodel_matter_power (float or array_like): matter power spectrum from halo model
    """
    return _vectorize_fn2(lib.halomodel_matter_power,
			  lib.halomodel_matter_power_vec, cosmo, k, a)

def halo_concentration(cosmo, halo_mass, a, label=2):
    """Halo concentration
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor
        halo_mass (float or array_like): mass of halo in Msun
    
    Returns:
        halo_concentration: measure of halo concentration
    """
    return _vectorize_fn4(lib.halo_concentration,
			  lib.halo_concentration_vec, cosmo, halo_mass, a, label)

