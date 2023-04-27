from __future__ import annotations

__all__ = ("BaryonsSchneider15",)

from numbers import Real
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

from .. import Pk2D, update
from . import Baryons

if TYPE_CHECKING:
    from .. import Cosmology


class BaryonsSchneider15(Baryons):
    r"""Baryonic correction model of :footcite:t:`Schneider15`.

    The boost factor :math:`f` is applied multiplicatively so that

    .. math::

        P_{\rm bar}(k, a) = P_{\rm nobar}(k, a) \, f(k, a).

    Refer to the `DESC Note
    <https://github.com/LSSTDESC/CCL/blob/master/doc/0000-ccl_note/>`_
    for details (needs compilation).

    References
    ----------
    .. footbibliography::

    Parameters
    ----------
    log10Mc
        Logarithmic mass scale of hot gas suppression.
    eta_b
        Ratio of escape to ejection radii (see linked publication).
    k_s
        Characteristic scale (wavenumber) of the stellar component, in units of
        :math:`\rm Mpc^{-1}`.

    Attributes
    ----------
    log10Mc

    eta_b

    k_s
    """
    __repr_attrs__ = __eq_attrs__ = ("log10Mc", "eta_b", "k_s")
    name = 'Schneider15'

    def __init__(
            self,
            *,
            log10Mc: Real = np.log10(1.2E14),
            eta_b: Real = 0.5,
            k_s: Real = 55.0
    ):
        self.log10Mc = log10Mc
        self.eta_b = eta_b
        self.k_s = k_s

    def boost_factor(
            self,
            cosmo: Cosmology,
            k: Union[Real, npt.NDArray[Real, 1]],
            a: Union[Real, npt.NDArray[Real, 1]],
    ) -> Union[float, npt.NDArray[float, 2]]:
        r"""Compute the baryonic boost factor.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k  : array_like (nk,)
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a : array_like (na,)
            Scale factor.

        Returns
        -------
        boost_factor : array_like (na, nk)
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

    @update(names=["log10Mc", "eta_b", "k_s"])
    def update_parameters(self) -> None:
        r"""Update the model parameters. All parameters this class accepts are
        updatable.
        """

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
