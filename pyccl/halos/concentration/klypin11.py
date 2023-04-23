__all__ = ("ConcentrationKlypin11",)

from ... import warn_api
from . import Concentration


class ConcentrationKlypin11(Concentration):
    r"""Concentration-mass relation by `Klypin et al. (2011)
    <https://arxiv.org/abs/1002.3660>`_. Only valid for S.O. masses with
    :math:`\Delta_{\rm vir}`.

    The concentration takes the form

    .. math::

        c(M_{\rm vir}) = 9.60 \left(
            \frac{M_{\rm vir}}{10^{12} h^{-1} \, {\rm M_\odot}}
            \right)^{-0.075},

    for distrinct halos.

    Parameters
    ---------
    mass_def : :class:`~pyccl.halos.MassDef` or str, fixed
        Mass definition for this :math:`c(M)` parametrization.
        It is fixed to :math:`\Delta_{\rm vir}`.
    """
    name = 'Klypin11'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def="vir"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "vir"

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo["h"] * 1E-12
        return 9.6 * (M * M_pivot_inv)**(-0.075)
