from . import ccllib as lib
import numpy as np
from .core import check


def nfw_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D NFW halo profile at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:obj:`Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    status = 0
    scalar = True if np.ndim(r) == 0 else False

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
    check(status, cosmo)
    if scalar:
        return rho_r[0]
    return rho_r


def nfw_profile_2d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 2D projected NFW halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:obj:`Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 2D projected NFW density at r,
         in units of Msun/Mpc^2.
    """
    status = 0
    scalar = True if np.ndim(r) == 0 else False

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
    check(status, cosmo)
    if scalar:
        return sigma_r[0]
    return sigma_r


def einasto_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D Einasto halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.
    The alpha parameter is calibrated using the relation with peak height in
    https://arxiv.org/pdf/1401.1216.pdf eqn5, assuming virial mass.

    Args:
        cosmo (:obj:`Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    # needed for part of the parameters
    cosmo.compute_sigma()

    status = 0
    scalar = True if np.ndim(r) == 0 else False

    # Convert to array if it's not already an array
    if not isinstance(r, np.ndarray):
        r = np.array([r, ]).flatten()

    nr = len(r)

    cosmo = cosmo.cosmo
    # Call function
    rho_r, status = lib.halo_profile_einasto_vec(
        cosmo, concentration, halo_mass,
        odelta, a, r, nr, status)

    # Check status and return
    check(status, cosmo)
    if scalar:
        return rho_r[0]
    return rho_r


def hernquist_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D Hernquist halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:obj:`Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    status = 0
    scalar = True if np.ndim(r) == 0 else False

    # Convert to array if it's not already an array
    if not isinstance(r, np.ndarray):
        r = np.array([r, ]).flatten()

    nr = len(r)

    cosmo = cosmo.cosmo
    # Call function
    rho_r, status = lib.halo_profile_hernquist_vec(
        cosmo, concentration, halo_mass,
        odelta, a, r, nr, status)

    # Check status and return
    check(status, cosmo)
    if scalar:
        return rho_r[0]
    return rho_r
