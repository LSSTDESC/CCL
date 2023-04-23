__all__ = ("BaryonsSchneider15",)

import numpy as np

from .. import Pk2D
from . import Baryons


class BaryonsSchneider15(Baryons):
    r"""Baryonic correction model of `Schneider & Teyssier (2015)
    <https://arxiv.org/abs/1510.06034>`_.

    The boost factor :math:`f` is applied multiplicatively so that

    .. math::

        P_{\rm bar}(k, a) = P_{\rm nobar}(k, a) \, f(k, a).

    Refer to the `DESC Note
    <https://github.com/LSSTDESC/CCL/blob/master/doc/0000-ccl_note/>`_
    for details (needs compilation).

    Parameters
    ----------
    log10Mc : float, optional
        Logarithmic mass scale of hot gas suppression.
        The default is :math:`\log_{10}\left(1.2 \times 10^{14} \right)`.
    eta_b : float, optional
        Ratio of escape to ejection radii (see linked publication).
        The default is :math:`0.5`.
    k_s : float, optional
        Characteristic scale (wavenumber) of the stellar component, in units of
        :math:`\rm Mpc^{-1}`. The default is :math:`55.0`.

    Attributes
    ----------
    name : str
        Name of the model, used when instantiating from the base class.
    log10Mc

    eta_b

    k_s
    """
    __repr_attrs__ = __eq_attrs__ = ("log10Mc", "eta_b", "k_s")
    name = 'Schneider15'

    def __init__(self, log10Mc=np.log10(1.2E14), eta_b=0.5, k_s=55.0):
        self.log10Mc = log10Mc
        self.eta_b = eta_b
        self.k_s = k_s

    def boost_factor(self, cosmo, k, a):
        r"""Compute the baryonic boost factor.

        Arguments
        ---------
        cosmo : :class:`~pyccl.Cosmology`
            Cosmological parameters.
        k : float or (nk,) array_like
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : float or (na,) array_like
            Scale factor.

        Returns
        -------
        boost_factor : float or (nk,) ``numpy.ndarray``
            Baryonic boost multiplicative factor.
        """
        a_use, k_use = map(np.atleast_1d, [a, k])
        a_use, k_use = a_use[:, None], k_use[None, :]

        z = 1/a_use - 1
        kh = k_use / cosmo['h']
        b0 = 0.105*self.log10Mc - 1.27
        bfunc = b0 / (1. + (z/2.3)**2.5)
        kg = 0.7 * (1-bfunc)**4 * self.eta_b**(-1.6)
        gf = bfunc / (1 + (kh/kg)**3) + 1. - bfunc
        scomp = 1 + (kh / self.k_s)**2
        fka = gf * scomp

        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, log10Mc=None, eta_b=None, k_s=None):
        r"""Update the model parameters. All parameters in :meth:`__init__`
        can be updated.
        """
        if log10Mc is not None:
            self.log10Mc = log10Mc
        if eta_b is not None:
            self.eta_b = eta_b
        if k_s is not None:
            self.k_s = k_s

    def _include_baryonic_effects(self, cosmo, pk):
        # Apply boost factor.
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        fka = self.boost_factor(cosmo, np.exp(lk_arr), a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
