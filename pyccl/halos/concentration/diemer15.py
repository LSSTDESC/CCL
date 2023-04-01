from ...base import warn_api
from ..halo_model_base import Concentration
import numpy as np


__all__ = ("ConcentrationDiemer15",)


class ConcentrationDiemer15(Concentration):
    r"""Concentration-mass relation by Diemer & Kravtsov (2015)
    :arXiv:1407.4730. Valid only for S.O. :math:`\Delta = 200c`
    mass definitions.

    The concentration takes the form

    .. math::

        c_{\rm 200c}(M, z) = \frac{c_{\rm min}}{2} \left[
            \left( \frac{\nu}{\nu_{\rm min}} \right)^{-\alpha}
            + \left( \frac{\nu}{\nu_{\rm min}} \right)^\beta \right],

    where :math:`c_{\rm min}` and :math:`\nu_{\rm min}` take the functional
    form :math:`X_{\rm min} = \chi_0 + \chi_1 n`. :math:`n` is the slope
    of the power spectrum, and :math:`(\chi_0,\chi_1)` are fitting parameters.

    .. note::

        The mass definition for this concentration is fixed to :math:`200c`.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`c(M)` parametrization.
        The default is :math:`\Delta=200c`.
    """
    name = 'Diemer15'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def="200c"):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.kappa = 1.0
        self.phi_0 = 6.58
        self.phi_1 = 1.27
        self.eta_0 = 7.28
        self.eta_1 = 1.56
        self.alpha = 1.08
        self.beta = 1.77

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"

    def _concentration(self, cosmo, M, a):
        # Compute power spectrum slope
        R = cosmo.mass2radius_lagrangian(M)
        k_R = 2.0 * np.pi / R * self.kappa

        cosmo.compute_linear_power()
        pk = cosmo.get_linear_power()
        n = pk(k_R, a, derivative=True)

        sig = cosmo.sigmaM(M, a)
        delta_c = 1.68647
        nu = delta_c / sig

        floor = self.phi_0 + n * self.phi_1
        nu0 = self.eta_0 + n * self.eta_1
        c = 0.5 * floor * ((nu0 / nu)**self.alpha +
                           (nu / nu0)**self.beta)
        return c
