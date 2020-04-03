from . import halos as hal
from .pyutils import deprecated


@deprecated(hal.HaloProfileNFW)
def nfw_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D NFW halo profile at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    .. note:: Note that this function is deprecated. Please use the
              functionality in the :mod:`~pyccl.halos.profiles` module.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration,
                                  mdef=mdef)
    p = hal.HaloProfileNFW(c, truncated=False)
    return p.real(cosmo, r, halo_mass, a, mdef)


@deprecated(hal.HaloProfileEinasto)
def einasto_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D Einasto halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.
    The alpha parameter is calibrated using the relation with peak height in
    https://arxiv.org/pdf/1401.1216.pdf eqn5, assuming virial mass.

    .. note:: Note that this function is deprecated. Please use the
              functionality in the :mod:`~pyccl.halos.profiles` module.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration,
                                  mdef=mdef)
    mdef = hal.MassDef(odelta, 'matter',
                       c_m_relation=c)
    p = hal.HaloProfileEinasto(c, truncated=False)
    return p.real(cosmo, r, halo_mass, a, mdef)


@deprecated(hal.HaloProfileHernquist)
def hernquist_profile_3d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 3D Hernquist halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    .. note:: Note that this function is deprecated. Please use the
              functionality in the :mod:`~pyccl.halos.profiles` module.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 3D NFW density at r, in units of Msun/Mpc^3.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration,
                                  mdef=mdef)
    p = hal.HaloProfileHernquist(c, truncated=False)
    return p.real(cosmo, r, halo_mass, a, mdef)


@deprecated(hal.HaloProfileNFW)
def nfw_profile_2d(cosmo, concentration, halo_mass, odelta, a, r):
    """Calculate the 2D projected NFW halo profile
    at a given radius or an array of radii,
    for a halo with a given mass, mass definition, and concentration,
    at a given scale factor, with a cosmology dependence.

    .. note:: Note that this function is deprecated. Please use the
              functionality in the :mod:`~pyccl.halos.profiles` module.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): cosmological parameters.
        concentration (float): halo concentration.
        halo_mass (float): halo masses; in units of Msun.
        odelta (float): overdensity with respect to mean matter density.
        a (float): scale factor.
        r (float or array_like): radius or radii to calculate profile for,
         in units of Mpc.

    Returns:
        float or array_like: 2D projected NFW density at r, \
         in units of Msun/Mpc^2.
    """
    mdef = hal.MassDef(odelta, 'matter')
    c = hal.ConcentrationConstant(c=concentration,
                                  mdef=mdef)
    p = hal.HaloProfileNFW(c, truncated=False,
                           projected_analytic=True)
    return p.projected(cosmo, r, halo_mass, a, mdef)
