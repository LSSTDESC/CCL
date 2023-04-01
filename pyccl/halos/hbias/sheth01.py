from ...base import warn_api
from ..halo_model_base import HaloBias


__all__ = ("HaloBiasSheth01",)


class HaloBiasSheth01(HaloBias):
    r"""Halo bias relation by Sheth et al. (2001) :arXiv:astro-ph/9907024.
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
    """

    name = "Sheth01"

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a = 0.707
        self.sqrta = 0.84083292038
        self.b = 0.5
        self.c = 0.6
        self.dc = 1.68647

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc/sigM
        anu2 = self.a * nu**2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1 + (self.sqrta * anu2 * (1 + self.b / anu2c) -
                    anu2c / (anu2c + t1)) / (self.sqrta * self.dc)
