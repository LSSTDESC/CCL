from .. import ccllib as lib
from ..core import check
from ..background import species_types
from ..base import CCLAutoRepr, CCLNamedClass, warn_api, deprecate_attr
from .halo_model_base import create_instance
import numpy as np


__all__ = ("mass2radius_lagrangian", "convert_concentration", "MassDef",
           "MassDef200m", "MassDef200c", "MassDef500c", "MassDefVir",
           "MassDefFof",)


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
    __repr_attrs__ = ("name",)
    __getattr__ = deprecate_attr(pairs=[('c_m_relation', 'concentration')]
                                 )(super.__getattribute__)

    @warn_api(pairs=[("c_m_relation", "concentration")])
    def __init__(self, Delta, rho_type=None, *, concentration=None):
        # Check it makes sense
        if isinstance(Delta, str) and Delta not in ["fof", "vir"]:
            raise ValueError(f"Unknown Delta type {Delta}.")
        if isinstance(Delta, (int, float)) and Delta < 0:
            raise ValueError("Delta must be a positive number.")
        if rho_type not in ['matter', 'critical']:
            raise ValueError("rho_type must be either ['matter'|'critical].'")

        self.Delta = Delta
        self.rho_type = rho_type
        self.species = species_types[rho_type]
        # c(M) relation
        if concentration is None:
            self.concentration = None
        else:
            from .concentration import Concentration
            self.concentration = Concentration.create_instance(
                concentration, mass_def=self)

    @property
    def name(self):
        """Give a name to this mass definition."""
        if isinstance(self.Delta, (int, float)):
            return f"{self.Delta}{self.rho_type[0]}"
        return f"{self.Delta}"

    def __eq__(self, other):
        # TODO: Remove after #1033 is merged.
        if type(self) != type(other):
            return False
        return self.name == other.name

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

    def _get_concentration(self, cosmo, M, a):
        """ Returns concentration for this mass definition.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: halo concentration.
        """
        if self.concentration is None:
            raise AttributeError("mass_def has no associated concentration.")
        return self.concentration.get_concentration(cosmo, M, a)

    @warn_api(pairs=[("mdef_other", "mass_def_other")])
    def translate_mass(self, cosmo, M, a, *, mass_def_other):
        """ Translate halo mass in this definition into another definition

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:obj:`MassDef`): another mass definition.

        Returns:
            float or array_like: halo masses in new definition.
        """
        if self == mass_def_other:
            return M
        if self.concentration is None:
            raise AttributeError("mass_def has no associated concentration.")
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
        """ Return mass definition subclass from name string.

        Args:
            name (string):
                a mass definition name (e.g. '200m' for Delta=200 matter)

        Returns:
            MassDef subclass corresponding to the input name.
        """
        try:
            return eval(f"MassDef{name.capitalize()}")
        except NameError:
            raise ValueError(f"Mass definition {name} not implemented.")


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef200m(concentration='Duffy08'):
    r""":math:`\Delta = 200m` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef(200, 'matter', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef200c(concentration='Duffy08'):
    r""":math:`\Delta = 200c` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef(200, 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDef500c(concentration='Ishiyama21'):
    r""":math:`\Delta = 500m` mass definition.

    Args:
        c_m (string): concentration-mass relation.
    """
    return MassDef(500, 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDefVir(concentration='Klypin11'):
    r""":math:`\Delta = \rm vir` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef('vir', 'critical', concentration=concentration)


@warn_api(pairs=[('c_m', 'concentration')])
def MassDefFof(concentration=None):
    r""":math:`\Delta = \rm FoF` mass definition.

    Args:
        concentration (string): concentration-mass relation.
    """
    return MassDef('fof', 'matter', concentration=concentration)
