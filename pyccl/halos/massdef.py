__all__ = ("mass2radius_lagrangian", "convert_concentration", "MassDef",
           "MassDef200m", "MassDef200c", "MassDef500c", "MassDefVir",
           "MassDefFof", "mass_translator",)

from functools import cached_property
from typing import Union, Callable

import numpy as np

from .. import CCLAutoRepr, CCLNamedClass, lib, check
from .. import warn_api, deprecate_attr
from . import Concentration, HaloBias, MassFunc


def mass2radius_lagrangian(cosmo, M):
    """ Returns Lagrangian radius for a halo of mass M.
    The lagrangian radius is defined as that enclosing
    the mass of the halo assuming a homogeneous Universe.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        M (float or array_like): halo mass in units of M_sun.

    Returns:
        float or array_like: lagrangian radius in comoving Mpc.
    """
    M_use = np.atleast_1d(M)
    R = (M_use / (4.18879020479 * cosmo.rho_x(1, 'matter')))**(1./3.)
    if np.ndim(M) == 0:
        return R[0]
    return R


@warn_api
def convert_concentration(cosmo, *, c_old, Delta_old, Delta_new):
    """ Computes the concentration parameter for a different mass definition.
    This is done assuming an NFW profile. The output concentration `c_new` is
    found by solving the equation:

    .. math::
        f(c_{\\rm old}) \\Delta_{\\rm old} = f(c_{\\rm new}) \\Delta_{\\rm new}

    where

    .. math::
        f(x) = \\frac{x^3}{\\log(1+x) - x/(1+x)}.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        c_old (float or array_like): concentration to translate from.
        Delta_old (float): Delta parameter associated to the input
            concentration. See description of the MassDef class.
        Delta_new (float): Delta parameter associated to the output
            concentration.

    Returns:
        float or array_like: concentration parameter for the new
        mass definition.
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
    """Halo mass definition. Halo masses are defined in terms of an overdensity
    parameter :math:`\\Delta` and an associated density :math:`X` (either the
    matter density or the critical density):

    .. math::
        M = \\frac{4 \\pi}{3} \\Delta\\,\\rho_X\\, R^3

    where :math:`R` is the halo radius. This object also holds methods to
    translate between :math:`R` and :math:`M`, and to translate masses between
    different definitions if a concentration-mass relation is provided.

    Args:
        Delta (float): overdensity parameter. Pass 'vir' if using virial
            overdensity.
        rho_type (string): either 'critical' or 'matter'.
        concentration (function, optional): concentration-mass relation.
            Provided as a `Concentration` object, or a string corresponding
            to one of the supported concentration-mass relations.
            If `None`, no c(M) relation will be attached to this mass
            definition (and hence one can't translate into other definitions).
    """
    __eq_attrs__ = ("name",)
    __getattr__ = deprecate_attr(pairs=[('c_m_relation', 'concentration')]
                                 )(super.__getattribute__)

    def __init__(self, Delta, rho_type=None):
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

    @cached_property
    def name(self):
        """Give a name to this mass definition."""
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def __repr__(self):
        return f"MassDef(Delta={self.Delta}, rho_type={self.rho_type})"

    def get_Delta(self, cosmo, a):
        """ Gets overdensity parameter associated to this mass
        definition.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            a (float): scale factor

        Returns:
            float : value of the overdensity parameter.
        """
        if self.Delta == 'fof':
            raise ValueError("FoF masses don't have an associated overdensity."
                             "Nor can they be translated into other masses")
        if self.Delta == 'vir':
            status = 0
            D, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
            return D
        return self.Delta

    def _get_Delta_m(self, cosmo, a):
        """ For SO-based mass definitions, this returns the corresponding
        value of Delta for a rho_matter-based definition.
        """
        delta = self.get_Delta(cosmo, a)
        if self.rho_type == 'matter':
            return delta
        om_this = cosmo.omega_x(a, self.rho_type)
        om_matt = cosmo.omega_x(a, 'matter')
        return delta * om_this / om_matt

    def get_mass(self, cosmo, R, a):
        """ Translates a halo radius into a mass

        .. math::
            M = \\frac{4 \\pi}{3} \\Delta\\,\\rho_X\\, R^3

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            R (float or array_like): halo radius in units of Mpc (physical, not
                comoving).
            a (float): scale factor.

        Returns:
            float or array_like: halo mass in units of M_sun.
        """
        R_use = np.atleast_1d(R)
        Delta = self.get_Delta(cosmo, a)
        M = 4.18879020479 * cosmo.rho_x(a, self.rho_type) * Delta * R_use**3
        if np.ndim(R) == 0:
            return M[0]
        return M

    def get_radius(self, cosmo, M, a):
        """ Translates a halo mass into a radius

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: halo radius in units of Mpc (physical, not
                comoving).
        """
        M_use = np.atleast_1d(M)
        Delta = self.get_Delta(cosmo, a)
        R = (M_use / (4.18879020479 * Delta *
                      cosmo.rho_x(a, self.rho_type)))**(1./3.)
        if np.ndim(M) == 0:
            return R[0]
        return R

    @classmethod
    def from_name(cls, name):
        """ Return mass definition subclass from name string.

        Args:
            name (string):
                a mass definition name (e.g. '200m' for Delta=200 matter)

        Returns:
            MassDef subclass corresponding to the input name.
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
        return lambda: cls(Delta, rho_type)  # noqa

    @classmethod
    def from_specs(cls, mass_def=None, *,
                   mass_function=None, halo_bias=None, concentration=None):
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


def MassDef200m():
    r""":math:`\Delta = 200m` mass definition."""
    return MassDef(200, 'matter')


def MassDef200c():
    r""":math:`\Delta = 200c` mass definition."""
    return MassDef(200, 'critical')


def MassDef500c():
    r""":math:`\Delta = 500m` mass definition."""
    return MassDef(500, 'critical')


def MassDefVir():
    r""":math:`\Delta = \rm vir` mass definition."""
    return MassDef('vir', 'critical')


def MassDefFof():
    r""":math:`\Delta = \rm FoF` mass definition."""
    return MassDef('fof', 'matter')


def mass_translator(*,
                    mass_in: Union[str, MassDef],
                    mass_out: Union[str, MassDef],
                    concentration: Union[str, Concentration]) -> Callable:
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
