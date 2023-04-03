from .. import ccllib as lib
from ..core import check
from ..background import species_types
from ..base import CCLAutoRepr, CCLNamedClass, warn_api, deprecate_attr
import numpy as np


__all__ = ("mass2radius_lagrangian", "convert_concentration", "MassDef",
           "MassDef200m", "MassDef200c", "MassDef500c", "MassDefVir",
           "MassDefFof",)


def mass2radius_lagrangian(cosmo, M):
    r"""Compute the Lagrangian radius of a halo.

    Defined as the radius enclosing the mass of the halo, assuming a
    homogeneous Universe

    .. math::

        R = \left( \frac{3M}{4\pi\rho_{0,\rm m}} \right)^\frac{1}{3},

    where :math:`\rho_{0,\rm m}` is the density of matter in the Universe
    today.

    Parameters
    ----------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    M : float or (nM,) array_like
        Halo mass in :math:`\rm M_\odot`.

    Returns
    -------
    radius : float or (nM,) ``numpy.ndarray``
        Lagrangian radius in comoving :math:`\rm Mpc`.
    """
    M_use = np.atleast_1d(M)
    R = (M_use / (4.18879020479 * cosmo.rho_x(1, 'matter')))**(1/3)
    if np.ndim(M) == 0:
        return R[0]
    return R


@warn_api
def convert_concentration(cosmo, *, c_old, Delta_old, Delta_new):
    r"""Convert the concentration to another overdensity parameter.

    The new concentration is found by solving the equation

    .. math::

        f(c_{\rm new}) = \frac{\Delta_{\rm old}}{\Delta_{\rm new}} \,
        f(c_{\rm new})

    where

    .. math::

        f(x) = \frac{x^3}{\ln(1+x) - x/(1+x)}

    is the NFW form factor.

    Arguments
    ---------
    c_old : float or (nc,) array_like
        Concentration to translate.
    Delta_old, Delta_new : float
        Overdensity (:math:`\Delta`) parameters associated with the
        halo mass definition of the old and new concentrations, respectively.
        See :class:`~pyccl.halos.massdef.MassDef` for details.

    Returns
    -------
    c_new : float or (nc,) ``numpy.ndarray``
        Concentration expressed in terms of the new overdensity parameter.
    """
    status = 0
    c_old_use = np.atleast_1d(c_old)
    c_new, status = lib.convert_concentration_vec(cosmo.cosmo,
                                                  Delta_old, c_old_use,
                                                  Delta_new, c_old_use.size,
                                                  status)
    check(status, cosmo=cosmo)

    if np.isscalar(c_old):
        return c_new[0]
    return c_new


class MassDef(CCLAutoRepr, CCLNamedClass):
    r"""Halo mass definition.

    Halo masses are defined in terms of an overdensity parameter :math:`\Delta`
    and a reference density type: either ``'matter'`` or ``'critical'``. The
    mass is related to the mass definition parameters via

    .. math::

        M = \frac{4 \pi}{3} \Delta \, \rho_{\rm X} \, R^3,

    where :math:`R` is the halo radius.

    Parameters
    ----------
    Delta : float, int-castable str, or {'fof', 'vir'}
        Spherical overdensity (S.O.) parameter. ``'fof'`` for friends-of-
        friends masses and ``'vir'`` for Virial masses.
    rho_type : {'critical', 'matter'}
        Reference mean density type.
    concentration : :class:`~pyccl.halos.Concentration`, str, or None, optional
        Concentration-mass relation. Provided either as a name string,
        or as a ``Concentration`` object. If ``None``, the mass definition
        cannot be translated to other mass definitions.
        The default is ``None``.

    Attributes
    ----------
    Delta : float or {'fof', 'vir'}
        S.O. parameter.
    rho_type : {'critical', 'matter'}
        Reference mean density type.
    concentration : :class:`~pyccl.halos.Concentration`
        Concentration-mass relation.
    name : str
        Short name of the mass definition, e.g. ``'200m'`` for
        ``(Delta, rho_type) == (200, 'matter')``.
    """
    __repr_attrs__ = ("name",)
    __getattr__ = deprecate_attr(pairs=[('c_m_relation', 'concentration')]
                                 )(super.__getattribute__)

    @warn_api(pairs=[("c_m_relation", "concentration")])
    def __init__(self, Delta, rho_type, *, concentration=None):
        # Check it makes sense
        if isinstance(Delta, str) and Delta.isdigit():
            Delta = int(Delta)
        if isinstance(Delta, str) and Delta not in ["fof", "vir"]:
            raise ValueError(f"Unknown Delta type {Delta}.")
        if isinstance(Delta, (int, float)) and Delta < 0:
            raise ValueError("Delta must be a positive number.")
        if rho_type not in ['matter', 'critical']:
            raise ValueError("rho_type must be either ['matter', 'critical].'")

        self.Delta = Delta
        self.rho_type = rho_type
        self._species = species_types[rho_type]
        # c(M) relation
        if concentration is None:
            self.concentration = None
        else:
            from .concentration import Concentration
            self.concentration = Concentration.initialize_from_input(
                concentration, mass_def=self)

    @property
    def name(self):
        r"""Name of the mass definition.

        If ``Delta`` is ``{'fof', 'vir'}`` just this is used.
        If it is a number, it is appended by the first letter of the reference
        density type, ``{'c', 'm'}``, as conventionally denoted.
        """
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def __eq__(self, other):
        # TODO: Remove after #1033 is merged.
        if type(self) != type(other):
            return False
        return self.name == other.name

    def get_Delta(self, cosmo, a):
        r"""Compute the overdensity parameter for this mass definition.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        a : float
            Scale factor.

        Returns
        -------
        Delta : float
            Overdensity parameter at ``a``.

        Raises
        ------
        ValueError
            If the mass definition is FoF, which do not have an associated
            overdensity parameter.
        """
        if self.Delta == 'fof':
            raise ValueError("FoF masses have no associated overdensity "
                             "and can't be translated into other masses.")
        if self.Delta == 'vir':
            status = 0
            D, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
            check(status)
            return D
        return self.Delta

    def get_Delta_matter(self, cosmo, a):
        r"""Compute the corresponding overdensity parameter for a
        :math:`\rho_{\rm m}`-based mass definition, :math:`\Delta_{\rm m}`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        a : float
            Scale factor.

        Returns
        -------
        Delta_m : float
            Overdensity parameter at ``a``.
        """
        delta = self.get_Delta(cosmo, a)
        if self.rho_type == 'matter':
            return delta
        om_this = cosmo.omega_x(a, self.rho_type)
        om_matt = cosmo.omega_x(a, 'matter')
        return delta * om_this / om_matt

    def get_mass(self, cosmo, R, a):
        r"""Translate halo radius to halo mass.

        .. math::

            M = \frac{4 \pi}{3} \Delta \,\rho_{\rm X} \, R^3

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        R: float or (nR,) array_like
            Halo radius in physical :math:`\rm Mpc`.
        a : float
            Scale factor.

        Returns
        -------
        mass : float or (nR,) ``numpy.ndarray``
            Halo mass in physical :math:`\rm M_\odot`.
        """
        R_use = np.atleast_1d(R)
        Delta = self.get_Delta(cosmo, a)
        M = 4.18879020479 * cosmo.rho_x(a, self.rho_type) * Delta * R_use**3
        if np.ndim(R) == 0:
            return M[0]
        return M

    def get_radius(self, cosmo, M, a):
        r"""Translate halo mass to halo radius.

        .. math::

            R = \left( \frac{3M}{4\pi \, \rho_{\rm X}(a)} \right)^\frac{1}{3},

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float
            Scale factor.

        Returns
        -------
        mass : float or (nM,) ``numpy.ndarray``
            Halo radius in physical :math:`\rm Mpc`.
        """
        M_use = np.atleast_1d(M)
        Delta = self.get_Delta(cosmo, a)
        R = (M_use / (4.18879020479 * Delta *
                      cosmo.rho_x(a, self.rho_type)))**(1/3)
        if np.ndim(M) == 0:
            return R[0]
        return R

    @warn_api(pairs=[("mdef_other", "mass_def_other")])
    def translate_mass(self, cosmo, M, a, *, mass_def_other):
        r"""Translate halo mass in this definition into another definition.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : float
            Scale factor.
        mass_def_other : :class:`~pyccl.halos.massdef.MassDef`
            Mass definition to translate to.

        Returns
        -------
        M_translated : float or (nM,) ndarray
            Halo masses in new definition in units of :math:`\rm M_\odot`.

        Raises
        ------
        AttributeError
            If the mass definition has no associated concentration.
        """
        if self == mass_def_other:
            return M
        if self.concentration is None:
            raise AttributeError("Mass definition has no associated c(M).")
        om_this = cosmo.omega_x(a, self.rho_type)
        D_this = self.get_Delta(cosmo, a) * om_this
        c_this = self._get_concentration(cosmo, M, a)
        R_this = self.get_radius(cosmo, M, a)
        om_new = cosmo.omega_x(a, mass_def_other.rho_type)
        D_new = mass_def_other.get_Delta(cosmo, a) * om_new
        c_new = convert_concentration(cosmo, c_old=c_this,
                                      Delta_old=D_this,
                                      Delta_new=D_new)
        R_new = c_new * R_this / c_this
        return mass_def_other.get_mass(cosmo, R_new, a)

    @classmethod
    def from_name(cls, name):
        r"""Return a mass definition factory from a name string.

        Arguments
        ---------
        name : str
            Name of the mass definition
            (e.g. ``'200m'`` for :math:`\Delta_{200{\rm m}}`).

        Returns
        -------
        mass_def_factory : callable
            Factory for the mass definition of the input name. The factory
            will be concentration-agnostic (i.e. the default value for it
            will be ``None``), unless the implementation already exists in CCL.

        Raises
        ------
        ValueError
            If the input string cannot be parsed.
        """
        MassDefName = f"MassDef{name.capitalize()}"
        if MassDefName in globals():
            # MassDef is defined in one of the implementations below.
            return globals()[MassDefName]
        parser = {"c": "critical", "m": "matter"}
        if len(name) < 2 or name[-1] not in parser:
            # Bogus input - can't parse it.
            raise ValueError("Could not parse mass definition string.")
        Delta, rho_type = name[:-1], parser[name[-1]]
        return lambda cm=None: cls(Delta, rho_type, concentration=cm)  # noqa


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef200m(concentration='Duffy08'):
    r""":math:`\Delta_{200{\rm m}}` mass definition.

    Arguments
    ---------
    concentration : str
        Name of the concentration-mass relation.
        The default is ``'Duffy08'``.
    """
    return MassDef(200, 'matter', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef200c(concentration='Duffy08'):
    r""":math:`\Delta_{200{\rm c}}` mass definition.

    Arguments
    ---------
    concentration : str
        Name of the concentration-mass relation.
        The default is ``'Duffy08'``.
    """
    return MassDef(200, 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef500c(concentration='Ishiyama21'):
    r""":math:`\Delta_{500{\rm c}}` mass definition.

    Arguments
    ---------
    concentration : str
        Name of the concentration-mass relation.
        The default is ``'Ishiyama21'``.
    """
    return MassDef(500, 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDefVir(concentration='Klypin11'):
    r""":math:`\Delta_{\rm vir}` mass definition.

    Arguments
    ---------
    concentration : str
        Name of the concentration-mass relation.
        The default is ``'Klypin11'``.
    """
    return MassDef('vir', 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDefFof(concentration=None):
    r""":math:`\Delta_{\rm FoF}` mass definition.

    Arguments
    ---------
    concentration : str
        This mass definition has no associated concentration.
        The default is ``None``.
    """
    return MassDef('fof', 'matter', concentration=concentration)
