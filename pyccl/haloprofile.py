"""
.. warning:: The functionality contained in this module has been deprecated
             in favour of the newer halo model implementation
             :py:mod:`~pyccl.halos`.
"""

__all__ = ("nfw_profile_3d", "einasto_profile_3d", "hernquist_profile_3d",
           "nfw_profile_2d",)

from .import deprecated
from . import halos as hal


@deprecated(hal.HaloProfileNFW)
def nfw_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D NFW halo profile at a given radius or an array
    of radii, for a halo with a given mass, mass definition, and
    concentration, at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmological parameters.
        concentration (:obj:`float`): halo concentration.
        halo_mass (:obj:`float`): halo masses; in units of Msun.
        odelta (:obj:`float`): overdensity with respect to mean matter density.
        a (:obj:`float`): scale factor.
        r (:obj:`float` or `array`): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        (:obj:`float` or `array`): 3D NFW density at r, in units of Msun/Mpc^3.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = hal.HaloProfileNFW(c, truncated=False)
    return p.real(cosmo, r, halo_mass, a)


@deprecated(hal.HaloProfileEinasto)
def einasto_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D Einasto halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.
    The alpha parameter is calibrated using the relation with peak height in
    https://arxiv.org/pdf/1401.1216.pdf eqn5, assuming virial mass.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmological parameters.
        concentration (:obj:`float`): halo concentration.
        halo_mass (:obj:`float`): halo masses; in units of Msun.
        odelta (:obj:`float`): overdensity with respect to mean matter density.
        a (:obj:`float`): scale factor.
        r (:obj:`float` or `array`): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        (:obj:`float` or `array`): 3D NFW density at r, in units of Msun/Mpc^3.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = hal.HaloProfileEinasto(c, truncated=False)
    return p.real(cosmo, r, halo_mass, a)


@deprecated(hal.HaloProfileHernquist)
def hernquist_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D Hernquist halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmological parameters.
        concentration (:obj:`float`): halo concentration.
        halo_mass (:obj:`float`): halo masses; in units of Msun.
        odelta (:obj:`float`): overdensity with respect to mean matter density.
        a (:obj:`float`): scale factor.
        r (:obj:`float` or `array`): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        (:obj:`float` or `array`): 3D NFW density at r, in units of Msun/Mpc^3.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = hal.HaloProfileHernquist(c, truncated=False)
    return p.real(cosmo, r, halo_mass, a)


@deprecated(hal.HaloProfileNFW)
def nfw_profile_2d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 2D projected NFW halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmological parameters.
        concentration (:obj:`float`): halo concentration.
        halo_mass (:obj:`float`): halo masses; in units of Msun.
        odelta (:obj:`float`): overdensity with respect to mean matter density.
        a (:obj:`float`): scale factor.
        r (:obj:`float` or `array`): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        (:obj:`float` or `array`): 2D projected NFW density at r, \
         in units of Msun/Mpc^2.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = hal.HaloProfileNFW(c, truncated=False, projected_analytic=True)
    return p.projected(cosmo, r, halo_mass, a)
