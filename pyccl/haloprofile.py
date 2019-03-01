from . import ccllib as lib
import numpy as np
from .core import check


def NFW_profile_3D(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D NFW halo profile at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:obj:`Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity parameter.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    status = 0
    scalar = True if isinstance(r, float) else False

    # Convert to array if it's not already an array
    if not isinstance(r, np.ndarray):
        r = np.array([r, ]).flatten()

    nr = len(r)

    cosmo = cosmo.cosmo
    # Call function
    rho_r, status = lib.halo_profile_nfw_vec(
        cosmo, concentration, halo_mass,
        odelta, a, r, nr, status)

    # Check status and return
    check(status)
    if scalar:
        return rho_r[0]
    return rho_r


def NFW_profile_2D(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 2D projected NFW halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:obj:`Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity parameter.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 2D projected NFW density at r,
         in units of Msun/Mpc^2.
    """
    status = 0
    scalar = True if isinstance(r, float) else False

    # Convert to array if it's not already an array
    if not isinstance(r, np.ndarray):
        r = np.array([r, ]).flatten()

    nr = len(r)

    cosmo = cosmo.cosmo
    # Call function
    sigma_r, status = lib.projected_halo_profile_nfw_vec(
        cosmo, concentration,
        halo_mass, odelta, a, r, nr, status)

    # Check status and return
    check(status)
    if scalar:
        return sigma_r[0]
    return sigma_r
