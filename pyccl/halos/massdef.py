from __future__ import annotations

__all__ = ("mass2radius_lagrangian", "convert_concentration", "MassDef",
           "MassDef200m", "MassDef200c", "MassDef500c", "MassDefVir",
           "MassDefFof", "mass_translator",)

import warnings
import weakref
from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING, Union, Callable

import numpy as np
import numpy.typing as npt

from .. import CCLDeprecationWarning, CCLNamedClass, CCLObject, lib, check
from .. import warn_api, deprecate_attr
from . import Concentration, HaloBias, MassFunc

if TYPE_CHECKING:
    from .. import Cosmology


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
    cosmo : :obj:`~pyccl.core.Cosmology`
        Cosmological parameters.
    M : int, float or (nM,) array_like
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
    c_old : int, float or (nc,) array_like
        Concentration to translate.
    Delta_old, Delta_new : int or float
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


class MassDef(CCLNamedClass, CCLObject):
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
    Delta : int, float or {'fof', 'vir'}
        S.O. parameter.
    rho_type : {'critical', 'matter'}
        Reference mean density type.
    concentration : :class:`~pyccl.halos.Concentration`
        Concentration-mass relation.
    name : str
        Short name of the mass definition, e.g. ``'200m'`` for
        ``(Delta, rho_type) == (200, 'matter')``.
    """
    __eq_attrs__ = ("name",)
    __getattr__ = deprecate_attr(pairs=[('c_m_relation', 'concentration')]
                                 )(super.__getattribute__)

    def __init__(self, Delta, rho_type, *, c_m_relation=None):
        # Check it makes sense
        if isinstance(Delta, str):
            if Delta.isdigit():
                Delta = int(Delta)
            elif Delta not in ["fof", "vir"]:
                raise ValueError(f"Unknown Delta type {Delta}.")
        if isinstance(Delta, (int, float)) and Delta < 0:
            raise ValueError("Delta must be a positive number.")
        if rho_type not in ['matter', 'critical']:
            raise ValueError("rho_type must be {'matter', 'critical'}.")

        self.Delta = Delta
        self.rho_type = rho_type

        # TODO: Remove c_m_relation for CCLv3.
        if c_m_relation is not None:
            warnings.warn("c_m_relation is deprecated from MassDef and will "
                          "be removed in CCLv3.0.0.", CCLDeprecationWarning)
            c_m_relation = Concentration.create_instance(
                c_m_relation, mass_def=weakref.proxy(self))
        self.concentration = c_m_relation

    @cached_property
    def name(self):
        r"""Name of the mass definition.

        If ``Delta`` is ``{'fof', 'vir'}`` just this is used.
        If it is a number, it is appended by the first letter of the reference
        density type, ``{'c', 'm'}``, as conventionally denoted.
        """
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def __repr__(self):
        return f"MassDef(Delta={self.Delta}, rho_type={self.rho_type})"

    def get_Delta(self, cosmo, a):
        r"""Compute the overdensity parameter for this mass definition.

        Arguments
        ---------
        cosmo : :obj:`~pyccl.core.Cosmology`
            Cosmological parameters.
        a : int or float
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
            raise ValueError("FoF masses don't have an associated overdensity."
                             "Nor can they be translated into other masses")
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
        cosmo : :obj:`~pyccl.core.Cosmology`
            Cosmological parameters.
        a : int or float
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
        cosmo : :obj:`~pyccl.core.Cosmology`
            Cosmological parameters.
        R: int, float or (nR,) array_like
            Halo radius in physical :math:`\rm Mpc`.
        a : int or float
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
        cosmo : :obj:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : int, float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : int or float
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
        cosmo : :obj:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : int, float or (nM,) array_like
            Halo mass in :math:`\rm M_\odot`.
        a : int or float
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
        # TODO: Remove for CCLv3.
        warnings.warn("translate_mass is a deprecated method of MassDef and "
                      "will be removed in CCLv3.0.0. Use `pyccl.halos.mass_"
                      "translator`.", CCLDeprecationWarning)
        if self == mass_def_other:
            return M
        if self.concentration is None:
            raise AttributeError("Mass definition has no associated c(M).")
        om_this = cosmo.omega_x(a, self.rho_type)
        D_this = self.get_Delta(cosmo, a) * om_this
        c_this = self.concentration(cosmo, M, a)
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
        # return cls(Delta, rho_type)  # TODO: Uncomment for CCLv3.
        return lambda: cls(Delta, rho_type)  # noqa  # TODO: Remove for CCLv3.

    # TODO: Uncomment for CCLv3 and remove CCLNamedClass inheritance.
    # create_instance = from_name

    @classmethod
    def from_specs(
            cls,
            mass_def: Union["MassDef", str, None] = None,
            *,
            mass_function: Union[MassFunc, str, None] = None,
            halo_bias: Union[HaloBias, str, None] = None,
            concentration: Union[Concentration, str, None] = None
    ):
        """Instantiate mass definition and halo model ingredients.

        Unspecified halo model ingredients are ignored. ``mass_def`` is always
        instantiated.

        Parameters
        ----------
        mass_def : MassDef, str or None, optional
            Mass definition. If a string, instantiate from its name. If None,
            obtain the one from the first specified halo model ingredient.
            The default is None.
        mass_function, halo_bias, concentration : \
            (MassFunc, HaloBias, Concentration), str or None, optional
            Halo model ingredients. Strings are auto-instantiated using
            ``mass_def``. None values are ignored. The defaults are None.

        Returns
        -------
        mass_def : MassDef

        mass_function : MassFunction, if specified

        halo_bias : HaloBias, if specified

        concentration : Concentration, if specified

        Raises
        ------
        ValueError
            If mass definition cannot be retrieved from halo model ingredients.
        ValueError
            If mass definitions are inconsistent.
        """
        values = mass_function, halo_bias, concentration
        idx = [value is not None for value in values]

        # Filter only the specified ones.
        values = np.array(values)[idx]
        names = np.array(["mass_function", "halo_bias", "concentration"])[idx]
        Types = np.array([MassFunc, HaloBias, Concentration])[idx]

        # Sanity check.
        if mass_def is None:
            for name, value in zip(names, values):
                if isinstance(value, str):
                    raise ValueError(f"Need mass_def if {name} is str.")

        # Instantiate mass_def.
        if mass_def is not None:
            mass_def = cls.create_instance(mass_def)  # instantiate directly
        else:
            mass_def = values[0].mass_def  # use the one in HMIngredients

        # Instantiate halo model ingredients.
        out = []
        for name, value, Type in zip(names, values, Types):
            instance = Type.create_instance(value, mass_def=mass_def)
            out.append(instance)

        # Check mass definition consistency.
        if out and set([x.mass_def for x in out]) != set([mass_def]):
            raise ValueError("Inconsistent mass definitions.")

        return mass_def, *out


# TODO: Remove these definitions and uncomment the new ones for CCLv3.
# These will all throw warnings now.
factory_warn = lambda: warnings.warn(  # noqa
    "In CCLv3.0.0 MassDef factories will become variables.",
    CCLDeprecationWarning)


def MassDef200m(c_m='Duffy08'):
    r""":math:`\Delta_{200{\rm m}}` mass definition.

    Arguments
    ---------
    c_m : str
        Name of the concentration-mass relation.
        The default is ``'Duffy08'``.
    """
    factory_warn()
    return MassDef(200, 'matter', c_m_relation=c_m)


def MassDef200c(c_m='Duffy08'):
    r""":math:`\Delta_{200{\rm c}}` mass definition.

    Arguments
    ---------
    c_m : str
        Name of the concentration-mass relation.
        The default is ``'Duffy08'``.
    """
    factory_warn()
    return MassDef(200, 'critical', c_m_relation=c_m)


def MassDef500c(c_m='Ishiyama21'):
    r""":math:`\Delta_{500{\rm c}}` mass definition.

    Arguments
    ---------
    c_m : str
        Name of the concentration-mass relation.
        The default is ``'Ishiyama21'``.
    """
    factory_warn()
    return MassDef(500, 'critical', c_m_relation=c_m)


def MassDefVir(c_m='Klypin11'):
    r""":math:`\Delta_{\rm vir}` mass definition.

    Arguments
    ---------
    c_m : str
        Name of the concentration-mass relation.
        The default is ``'Klypin11'``.
    """
    factory_warn()
    return MassDef('vir', 'critical', c_m_relation=c_m)


def MassDefFof():
    r""":math:`\Delta_{\rm FoF}` mass definition."""
    return MassDef("fof", "matter")


# MassDef200m = MassDef(200, "matter")
# MassDef200c = MassDef(200, "critical")
# MassDef500c = MassDef(500, "critical")
# MassDefVir = MassDef("vir", "critical")
# MassDefFof = MassDef("fof", "matter")


def mass_translator(
        *,
        mass_in: Union[str, MassDef],
        mass_out: Union[str, MassDef],
        concentration: Union[str, Concentration]
) -> Callable[[Cosmology, Union[Real, npt.NDarray], Real],
              Union[Real, npt.NDarray]]:
    """Translate between mass definitions, assuming an NFW profile."""

    mass_in = MassDef.create_instance(mass_in)
    mass_out = MassDef.create_instance(mass_out)
    concentration = Concentration.create_instance(concentration,
                                                  mass_def=mass_in)
    if concentration.mass_def != mass_in:
        raise ValueError("mass_def of concentration doesn't match mass_in")

    def translate(cosmo, M, a):
        if mass_in == mass_out:
            return M

        c_in = concentration(cosmo, M, a)
        Om_in = cosmo.omega_x(a, mass_in.rho_type)
        D_in = mass_in.get_Delta(cosmo, a) * Om_in
        R_in = mass_in.get_radius(cosmo, M, a)

        Om_out = cosmo.omega_x(a, mass_out.rho_type)
        D_out = mass_out.get_Delta(cosmo, a) * Om_out
        c_out = convert_concentration(
            cosmo, c_old=c_in, Delta_old=D_in, Delta_new=D_out)
        R_out = R_in * c_out/c_in
        return mass_out.get_mass(cosmo, R_out, a)

    return translate
