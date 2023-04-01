from ...base import warn_api
from ..halo_model_base import HaloBias


__all__ = ("HaloBiasBhattacharya11",)


class HaloBiasBhattacharya11(HaloBias):
    r"""Halo bias relation by Bhattacharya et al. (2011) :arXiv:1005.2239.
    Valid for FoF masses only.

    The halo bias takes the form

    .. math::

        b(M, z) = 1 + \frac{\tilde{\alpha}\nu - \tilde{q}}{\delta_{\rm c}}
        + \frac{2\tilde{p}/\delta_{\rm c}}{1+(\tilde{\alpha}\nu)^{\tilde{p}}},

    where :math:`\nu = \delta_{\rm c}^2 / \sigma^2`, and every parameter
    with a tilde derives from redshift via a power law of the form
    :math:`\tilde{x} = x_0 / (1 + z)^{\alpha_x}`, where :math:`x_0`
    and :math:`\alpha_x` are fitted parameters.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`b(M)` parametrization.
        The default is :math:`{\rm FoF}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.
    """
    name = "Bhattacharya11"

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = 1.68647

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1 + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc
