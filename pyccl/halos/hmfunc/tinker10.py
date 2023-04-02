from ...base import warn_api
from ..halo_model_base import MassFunc
import numpy as np
from scipy.interpolate import interp1d


__all__ = ("MassFuncTinker10",)


class MassFuncTinker10(MassFunc):
    r"""Halo mass function by Tinker et al. (2010) :arXiv:1001.3162.
    Valid for any S.O. masses with :math:`\Delta \in (200{\rm m},3200{\rm m})`.

    The mass function takes the form

    .. math::

        n(M, z) = \alpha \, \left[ 1 + (\beta\nu)^{-2\phi} \right] \,
        \nu^{2\eta} \, \exp{ \left( -\frac{\gamma\nu^2}{2} \right) },

    where :math:`\nu \equiv \delta_c/\sigma` is the peak height, and
    :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\eta`, :math:`\phi`
    follow time-dependent power laws of the form :math:`x = x_0 (1+z)^{x_z}`,
    with :math:`(x_0, x_z)` fitted parameters.

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`n(M)` parametrization.
        The default is :math:`200{\rm m}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.
    norm_all_z : bool, optional
        Whether the mass function is normalized at all :math:`z`. If False,
        it is only normalized at :math:`z=0`. The default is False.

    Raises
    ------
    ValueError
        Interpolation out of bounds. :math:`\Delta_m` for the particular
        combination of mass definition and scale factor is out of bounds with
        the range of the mass function.
    """
    __repr_attrs__ = ("mass_def", "mass_def_strict", "norm_all_z",)
    name = 'Tinker10'

    @warn_api
    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True,
                 norm_all_z=False):
        self.norm_all_z = norm_all_z
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        delta = np.array(
            [200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
        alpha = np.array(
            [0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327])
        beta = np.array(
            [0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702])
        gamma = np.array(
            [0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81])
        phi = np.array(
            [-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49])
        eta = np.array(
            [-0.243, -0.261, -0.261, -0.273,
             -0.278, -0.301, -0.301, -0.319, -0.336])

        ldelta = np.log10(delta)
        self.pA0 = interp1d(ldelta, alpha)
        self.pa0 = interp1d(ldelta, eta)
        self.pb0 = interp1d(ldelta, beta)
        self.pc0 = interp1d(ldelta, gamma)
        self.pd0 = interp1d(ldelta, phi)
        if self.norm_all_z:
            p = np.array(
                [-0.158, -0.195, -0.213, -0.254, -0.281,
                 -0.349, -0.367, -0.435, -0.504])
            q = np.array(
                [0.0128, 0.0128, 0.0143, 0.0154, 0.0172,
                 0.0174, 0.0199, 0.0203, 0.0205])
            self.pp0 = interp1d(ldelta, p)
            self.pq0 = interp1d(ldelta, q)

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = np.log10(self.mass_def._get_Delta_m(cosmo, a))
        nu = 1.686 / sigM
        # redshift evolution only up to z=3
        a = np.clip(a, 0.25, 1)
        pa = self.pa0(ld) * a**(-0.27)
        pb = self.pb0(ld) * a**(-0.20)
        pc = self.pc0(ld) * a**0.01
        pd = self.pd0(ld) * a**0.08
        pA0 = self.pA0(ld)
        if self.norm_all_z:
            z = 1/a - 1
            pp = self.pp0(ld)
            pq = self.pq0(ld)
            pA0 *= np.exp(z * (pp + pq * z))
        return nu * pA0 * (1 + (pb * nu)**(-2 * pd)) * (
            nu**(2 * pa) * np.exp(-0.5 * pc * nu**2))
