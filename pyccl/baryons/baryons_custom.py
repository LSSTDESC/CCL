__all__ = ("BaryonsCustom",)

import numpy as np

from .. import Pk2D
from . import Baryons
from scipy.interpolate import RegularGridInterpolator

class BaryonsCustom(Baryons):
    """The custom baryonic model boost factor for baryons. 

    The boost factor is applied multiplicatively so that
    :math:`P_{\\rm bar.}(k, a) = P_{\\rm DMO}(k, a)\\, f_{\\rm BCM}(k, a)`.

    Args:
        boost_data (:obj:`array`): Array containing the boost factor data.
        k_data (:obj:`array`): Wavenumber (in :math:`{\\rm Mpc}^{-1}`).
        a_data (:obj:`array`): Scale factor.
    """
    name = 'BaryonsCustom'
    __repr_attrs__ = __eq_attrs__ = ("boost_data", "k_data", "a_data")

    def __init__(self, boost_data, k_data, a_data):
        self.boost_data = boost_data
        self.k_data = k_data
        self.a_data = a_data

        # Create interpolator in (a, k) space
        self._interpolator = RegularGridInterpolator(
            points=(a_data, k_data),
            values=boost_data,
            bounds_error=False,
            fill_value=1.0  # default to 1 (no boost) outside bounds
        )

    def boost_factor(self, cosmo, k, a):
        """Interpolated baryonic boost factor.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
            k (:obj:`float` or `array`): Wavenumber (in :math:`{\\rm Mpc}^{-1}`).
            a (:obj:`float` or `array`): Scale factor.

        Returns:
            :obj:`float` or `array`: Correction factor to apply to \
            the power spectrum.
        """ # noqa
        a_arr = np.atleast_1d(a)
        k_arr = np.atleast_1d(k)
        a_mesh, k_mesh = np.meshgrid(a_arr, k_arr, indexing='ij')
        query_points = np.column_stack([a_mesh.ravel(), k_mesh.ravel()])

        boost_vals = self._interpolator(query_points).reshape(a_mesh.shape)
        
        # Match output shape to inputs
        if np.ndim(a) == 0 and np.ndim(k) == 0:
            return boost_vals[0, 0]
        elif np.ndim(a) == 0:
            return boost_vals[0]
        elif np.ndim(k) == 0:
            return boost_vals[:, 0]
        return boost_vals

    def update_parameters(self, boost_data=None, k_data=None, a_data=None):
        """Update BCM parameters. All parameters set to ``None`` will
        be left untouched.

        Args:
            log10Mc (:obj:`float`): logarithmic mass scale of hot
                gas suppression.
            eta_b (:obj:`float`): ratio of escape to ejection radii.
            k_s (:obj:`float`): Characteristic scale (wavenumber) of
                the stellar component.
        """
        if boost_data is not None:
            self.boost_data = boost_data
        if k_data is not None:
            self.k_data = k_data
        if a_data is not None:
            self.a_data = a_data

        # Update interpolator in (a, k) space
        if boost_data is not None or k_data is not None or a_data is not None:
            self._interpolator = RegularGridInterpolator(
                points=(a_data, k_data),
                values=boost_data,
                bounds_error=False,
                fill_value=1.0  # default to 1 (no boost) outside bounds
            )

    def _include_baryonic_effects(self, cosmo, pk):
        # Applies boost factor
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        fka = self.boost_factor(cosmo, k_arr, a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
