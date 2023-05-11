r"""
==============================
Tracers (:mod:`pyccl.tracers`)
==============================

Tracers of the large-scale structure.

Tracers represent projected quantities which can be cross-correlated to compute
angular power spectra (see :func:`~pyccl.cells.angular_cl`). In the most
general case, the angular power spectrum between two tracers is given by

.. math::

    C^{\alpha\beta}_\ell = \frac{2}{\pi} \int {\rm d}\chi_1 \, {\rm d}\chi_2 \,
    {\rm d}k \, k^2 \, P_{\alpha\beta}(k, \chi_1, \chi_2) \,
    \Delta^\alpha_\ell(k, \chi_1) \, \Delta^\beta_\ell(k, \chi_2),

where :math:`P_{\alpha\beta}` is a generalized power spectrum (see
:class:`~pyccl.pk2d.Pk2D`), and :math:`\Delta^\alpha_\ell(k,\chi)` is the sum
of different contributions associated to tracer :math:`\alpha`, where every
contribution takes the form

.. math::

    \Delta^\alpha_\ell(k, \chi) = f^\alpha_\ell \, W_\alpha(\chi) \,
    T_\alpha(k, \chi) \, j^{(n_\alpha)}_\ell(k\chi).

Here, :math:`f^\alpha_\ell` is an :math:`\ell`-dependent *prefactor*,
:math:`W_\alpha(\chi)` is the *radial kernel*, :math:`T_\alpha(k,\chi)` is
the *transfer function*, and :math:`j^{(n)}_\ell(x)` is a generalized version
of the *spherical Bessel functions*.

Descriptions of each of these ingredients, and how to implement generalised
tracers, can be found in the documentation of :class:`Tracer`.
"""

from __future__ import annotations

__all__ = ("get_density_kernel", "get_lensing_kernel", "get_kappa_kernel",
           "Tracer", "NzTracer", "NumberCountsTracer", "WeakLensingTracer",
           "CMBLensingTracer", "tSZTracer", "CIBTracer", "ISWTracer",)

import warnings
from numbers import Real
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import ccllib as lib
from .errors import CCLWarning
from ._core.parameters import physical_constants
from ._core import CCLObject, warn_api
from .pyutils import (_check_array_params, NoneArr, _vectorize_fn,
                      _get_spline1d_arrays, _get_spline2d_arrays, check)

if TYPE_CHECKING:
    from . import Cosmology


def _Sig_MG(
        cosmo: Cosmology,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Redshift-dependent modification to Poisson equation for massless
    particles under modified gravity.

    Assumed to be proportional to :math:`\Omega_\Lambda(z)`,
    see e.g. Abbott et al. 2018, 1810.02499, Eq. 9.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    k : array_like (nk,)
        Wavenumber in :math:`\rm Mpc`.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na, nk)
        MG modification to the Poisson equation.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.Sig_MG, lib.Sig_MG_vec, cosmo, x=a, x2=k)


def _check_background_spline_compatibility(cosmo: Cosmology,
                                           z: NDArray[Real]) -> None:
    """Check that redshift `z` is within the support of the background splines
    of `cosmo`.
    """
    cosmo.compute_distances()
    a_bg, _ = _get_spline1d_arrays(cosmo.cosmo.data.chi)
    a = 1/(1+z)

    if a.min() < a_bg.min() or a.max() > a_bg.max():
        raise ValueError(
            "Tracer has wider redshift support than internal CCL splines. "
            f"Tracer: z=[{1/a.max()-1}, {1/a.min()-1}]. "
            f"Background splines: z=[{1/a_bg.max()-1}, {1/a_bg.min()-1}].")


@warn_api
def get_density_kernel(
        cosmo: Cosmology,
        *,
        dndz: Tuple[NDArray[Real], NDArray[Real]]
) -> Tuple[NDArray[float], NDArray[float]]:
    r"""Get the radial kernel :math:`W(\chi)` for clustering tracers,

    .. math::

        W(\chi) = p(z) \, H(z),

    where :math:`p(z)` is the normalized redshift distribution.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    dndz
        The (unnormalized) redshift distribution (e.g. ``dndz=(z, nz)``).
        Units are arbitrary because it is internally normalized to integrate to
        :math:`1`.

    Returns
    -------

        :math:`\chi` and :math:`W(\chi)`.

    See Also
    --------
    :meth:`~Tracer.get_kernel` : Get all radial kernels of a tracer collection.
    """
    z_n, n = _check_array_params(dndz, 'dndz')
    _check_background_spline_compatibility(cosmo, dndz[0])
    # this call inits the distance splines neded by the kernel functions
    chi = cosmo.comoving_radial_distance(1./(1.+z_n))
    status = 0
    wchi, status = lib.get_number_counts_kernel_wrapper(cosmo.cosmo,
                                                        z_n, n,
                                                        len(z_n),
                                                        status)
    cosmo.check(status)
    return chi, wchi


@warn_api
def get_lensing_kernel(
        cosmo: Cosmology,
        *,
        dndz: Tuple[NDArray[Real], NDArray[Real]],
        mag_bias: Optional[Tuple[NDArray[Real], NDArray[Real]]] = None,
        n_chi: Optional[int] = None
) -> Tuple[NDArray[float], NDArray[float]]:
    r"""Get the radial kernel for weak lensing tracers,

    .. math::

        W(\chi) = \frac{1}{\chi} \int_{z(\chi)}^\infty {\rm d}z' \,
        \left( 1 - \frac{\chi}{\chi(z')} \right) \, p_\alpha(z').


    Arguments
    ---------
    cosmo
        Cosmological parameters
    dndz
        The (unnormalized) redshift distribution (e.g. ``dndz=(z, nz)``). Units
        are arbitrary because it is internally normalized to integrate to
        :math:`1`.
    mag_bias
        Magnification bias. If provided, the output is the magnification
        kernel. The default assumes :math:`s = 0`.
    n_chi
        Number of distance samples.

    Returns
    -------

        :math:`\chi` and :math:`W(\chi)`, the lensing shear kernel or the
        magnification kernel, depending on whether `mag_bias` is provided.

    See Also
    --------
    :meth:`~Tracer.get_kernel` : Get all radial kernels of a tracer collection.
    """
    # we need the distance functions at the C layer
    cosmo.compute_distances()

    z_n, n = _check_array_params(dndz, 'dndz')
    has_magbias = mag_bias is not None
    z_s, s = _check_array_params(mag_bias, 'mag_bias')
    _check_background_spline_compatibility(cosmo, dndz[0])

    if n_chi is None:
        # Calculate number of samples in chi
        n_chi = lib.get_nchi_lensing_kernel_wrapper(z_n)

    if (n_chi > len(z_n)
            and cosmo.cosmo.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION):
        warnings.warn(
            f"The number of samples in the n(z) ({len(z_n)}) is smaller than "
            f"the number of samples in the lensing kernel ({n_chi}). Consider "
            "disabling spline integration for the lensing kernel by setting "
            "pyccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False "
            "before instantiating the Cosmology passed.", category=CCLWarning)

    # Compute array of chis
    status = 0
    chi, status = lib.get_chis_lensing_kernel_wrapper(cosmo.cosmo, z_n[-1],
                                                      n_chi, status)
    # Compute kernel
    wchi, status = lib.get_lensing_kernel_wrapper(cosmo.cosmo,
                                                  z_n, n, z_n[-1],
                                                  int(has_magbias), z_s, s,
                                                  chi, n_chi, status)
    cosmo.check(status)
    return chi, wchi


@warn_api(pairs=[("nsamples", "n_samples")])
def get_kappa_kernel(
        cosmo: Cosmology,
        *,
        z_source: Real,
        n_samples: int = 100
) -> Tuple[NDArray[float], NDArray[float]]:
    r"""Get the radial kernel for CMB lensing tracers,

    .. math::

        W(\chi) = K_\ell \frac{3 H_0^2 \Omega_{\rm m}}{2a(\chi)} \,
        \chi \, \left( 1 - \frac{\chi}{\chi_{\rm CMB}} \right),

    where

    .. math::

        K_\ell = \frac{\ell(\ell + 1)}{(\ell + 1/2)^2},

    and :math:`\chi_{\rm CMB}` is the comoving radial distance to the
    last-scattering surface.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    z_source
        Redshift of the source plane.
    n_samples
        Number of equispaced samples in radial distance.
        O(100) is usually enough, as the kernel is smooth.

    Returns
    -------

        :math:`\chi` and :math:`W(\chi)`.

    See Also
    --------
    :meth:`~Tracer.get_kernel` : Get all radial kernels of a tracer collection.
    """
    _check_background_spline_compatibility(cosmo, np.array([z_source]))
    # this call inits the distance splines neded by the kernel functions
    chi_source = cosmo.comoving_radial_distance(1./(1.+z_source))
    chi = np.linspace(0, chi_source, n_samples)

    status = 0
    wchi, status = lib.get_kappa_kernel_wrapper(cosmo.cosmo, chi_source,
                                                chi, n_samples, status)
    check(status, cosmo=cosmo)
    return chi, wchi


class Tracer(CCLObject):
    r"""Tracer of the large-scale structure.

    Contains the necessary information to describe the contribution of
    observables to the power spectrum. Tracers are composed of four main
    ingredients:

    * A radial kernel, which expresses the support in redshift/distance
      over which this tracer extends.

    * A transfer function, which describes the connection between the tracer
      and the power spectrum on different scales and at different cosmic times.

    * An :math:`\ell`-dependent prefactor, associated with angular derivatives
      of some fundamental quantity.

    * The order of the derivative of the Bessel functions with which
      they enter the computation of the angular power spectrum.

    :class:`~Tracer` objects are collections of individual tracers, which are
    combined to calculate their total imprint on the power spectrum. Refer to
    Sec. 4.9 of the CCL note for details.
    """
    from ._core.repr_ import build_string_Tracer as __repr__

    def __init__(self):
        # Do nothing, just initialize list of tracers
        self._trc = []

    @property
    def chi_min(self) -> Union[float, None]:
        r""":math:`\chi_{\min}` if it exists; None otherwise. For more than one
        tracer in the collection, the lowest value is returned.
        """
        chis = [tr.chi_min for tr in self._trc]
        return min(chis) if chis else None

    @property
    def chi_max(self) -> Union[float, None]:
        r""":math:`\chi_{\max}` if it exists; None otherwise. For more than one
        tracer in the collection, the largest value is returned.
        """
        chis = [tr.chi_max for tr in self._trc]
        return max(chis) if chis else None

    def __eq__(self, other):
        # Check object id.
        if self is other:
            return True

        # Check the object class.
        if type(self) is not type(other):
            return False

        # If the tracer collections are empty, return early.
        if not (self or other):
            return True

        # If the tracer collections are not the same length, return early.
        if len(self._trc) != len(other._trc):
            return False

        # Check `der_angles` & `der_bessel` for each tracer in the collection.
        bessel = self.get_bessel_derivative(), other.get_bessel_derivative()
        angles = self.get_angles_derivative(), other.get_angles_derivative()
        if not (np.array_equal(*bessel) and np.array_equal(*angles)):
            return False

        # Check the kernels.
        for t1, t2 in zip(self._trc, other._trc):
            if bool(t1.kernel) ^ bool(t2.kernel):
                # only one of them has a kernel
                return False
            if t1.kernel is None:
                # none of them has a kernel
                continue
            if not np.array_equal(_get_spline1d_arrays(t1.kernel.spline),
                                  _get_spline1d_arrays(t2.kernel.spline)):
                # both have kernels, but they are unequal
                return False

        # Check the transfer functions.
        for t1, t2 in zip(self._trc, other._trc):
            if bool(t1.transfer) ^ bool(t2.transfer):
                # only one of them has a transfer
                return False
            if t1.transfer is None:
                # none of them has a transfer
                continue
            # Check the characteristics of the transfer function.
            for arg in ("extrap_order_lok", "extrap_order_hik",
                        "is_factorizable", "is_log"):
                if getattr(t1.transfer, arg) != getattr(t2.transfer, arg):
                    return False

            c2py = {"fa": _get_spline1d_arrays,
                    "fk": _get_spline1d_arrays,
                    "fka": _get_spline2d_arrays}
            for attr in c2py.keys():
                spl1 = getattr(t1.transfer, attr, None)
                spl2 = getattr(t2.transfer, attr, None)
                if bool(spl1) ^ bool(spl2):
                    # only one of them has this transfer type
                    return False
                if spl1 is None:
                    # none of them has this transfer type
                    continue
                # `pts` contain the the grid points and the transfer functions
                pts1, pts2 = c2py[attr](spl1), c2py[attr](spl2)
                for pt1, pt2 in zip(pts1, pts2):
                    # loop through output points of `_get_splinend_arrays`
                    if not np.array_equal(pt1, pt2):
                        # both have this transfer type, but they are unequal
                        # or are defined at different grid points
                        return False
        return True

    def __hash__(self):
        return hash(repr(self))

    def __bool__(self):
        return bool(self._trc)

    def get_kernel(
            self,
            chi: Optional[Real, NDArray[Real]] = None
    ) -> Union[List[NDArray[float]],
               Tuple[List[NDArray[float]],
                     List[NDArray[float]]]]:
        r"""Get the radial kernels for all tracers in this collection.

        Arguments
        ---------
        chi : array_like (nchi,)
            Comoving radial distance :math:`\chi` (in :math:`\rm Mpc`) to
            evaluate the kernels. If not provided, return the spline knots.

        Returns
        -------
        array_like (nchi,)
            List of radial kernels. If `chi` is not provided, return two lists:
            one of the internal spline knots for `chi` and one for the kernels.
        """
        if chi is None:
            chis = []
        else:
            chi_use = np.atleast_1d(chi)

        kernels = []
        for t in self._trc:
            if t.kernel is None:
                continue
            else:
                if chi is None:
                    chi_use, w = _get_spline1d_arrays(t.kernel.spline)
                    chis.append(chi_use)
                else:
                    status = 0
                    w, status = lib.cl_tracer_get_kernel(
                        t, chi_use, chi_use.size, status)
                    check(status)
                kernels.append(w)

        if chi is None:
            return kernels, chis
        kernels = np.array(kernels)
        if np.ndim(chi) == 0 and kernels.shape != (0,):
            kernels = np.squeeze(kernels, axis=-1)
        return kernels

    def get_f_ell(
            self,
            ell: Union[Real, NDArray[Real]]
    ) -> NDArray[float]:
        r"""Get the :math:`\ell`-dependent prefactors for all tracers in this
        collection.

        Arguments
        ---------
        ell : array_like (nell,)
            Multipole.

        Returns
        -------
        array_like (N_tracer, nell)
            Prefactors. `N_tracer` is the number of tracers in the collection.
        """
        ell_use = np.atleast_1d(ell)
        f_ells = []
        for t in self._trc:
            status = 0
            f, status = lib.cl_tracer_get_f_ell(t, ell_use,
                                                ell_use.size,
                                                status)
            check(status)
            f_ells.append(f)
        f_ells = np.array(f_ells)
        if np.ndim(ell) == 0:
            if f_ells.shape != (0,):
                f_ells = np.squeeze(f_ells, axis=-1)
        return f_ells

    def get_transfer(
            self,
            lk: Union[Real, NDArray[Real]],
            a: Union[Real, NDArray[Real]]
    ) -> NDArray[float]:
        r"""Get the transfer functions for all tracers in this collection.

        Arguments
        ---------
        lk : array_like (nk,)
            Natural logarithm of the wavenumber, :math:`\ln k`
            (in units of :math:`\rm Mpc`).
        a : array_like (na,)
            Scale factor.

        Returns
        -------
        array_like (N_tracer, nk, na)
            The transfer functions for each tracer. `N_tracer` is the number of
            tracers in the collection. Note the unusual return order for
            quantities `lk` and `a`.
        """
        lk_use = np.atleast_1d(lk)
        a_use = np.atleast_1d(a)
        transfers = []
        for t in self._trc:
            status = 0
            t, status = lib.cl_tracer_get_transfer(t, lk_use, a_use,
                                                   lk_use.size * a_use.size,
                                                   status)
            check(status)
            transfers.append(t.reshape([lk_use.size, a_use.size]))
        transfers = np.array(transfers)
        if transfers.shape != (0,):
            if np.ndim(a) == 0:
                transfers = np.squeeze(transfers, axis=-1)
                if np.ndim(lk) == 0:
                    transfers = np.squeeze(transfers, axis=-1)
            else:
                if np.ndim(lk) == 0:
                    transfers = np.squeeze(transfers, axis=-2)
        return transfers

    def get_bessel_derivative(self) -> NDArray[float]:
        """Get Bessel function derivative orders for all tracers in this
        colelction.

        Returns
        -------
        ndarray : (N_tracer,)
            Bessel derivative orders. `N_tracer` is the number of tracers in
            the collection.
        """
        return np.array([t.der_bessel for t in self._trc])

    def get_angles_derivative(self) -> NDArray[float]:
        r"""Get ``enum`` of the :math:`\ell`-dependent prefactor for all
        tracers contained in this tracer collection.
        """
        return np.array([t.der_angles for t in self._trc])

    def _MG_add_tracer(self, cosmo, kernel, z_b, der_bessel=0, der_angles=0,
                       bias_transfer_a=None, bias_transfer_k=None):
        """Set the modified gravity transfer function in the right format for
        different cases, including when IAs are present.
        """
        # Getting MG transfer function and building a k-array
        mg_transfer = self._get_MG_transfer_function(cosmo, z_b)

        # case with no astro biases
        if ((bias_transfer_a is None) and (bias_transfer_k is None)):
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a and  k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * (bias_transfer_k[1] *
                                mg_transfer[2]).T).T)
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on a but not k
        elif ((bias_transfer_a is not None) and (bias_transfer_k is None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_a[1] * mg_transfer[2].T).T)
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

        #  case of an astro bias depending on k but not a
        elif ((bias_transfer_a is None) and (bias_transfer_k is not None)):
            mg_transfer_new = (mg_transfer[0], mg_transfer[1],
                               (bias_transfer_k[1] * mg_transfer[2]))
            self.add_tracer(cosmo, kernel=kernel, transfer_ka=mg_transfer_new,
                            der_bessel=der_bessel, der_angles=der_angles)

    def _get_MG_transfer_function(
            self,
            cosmo: Cosmology,
            z: Union[Real, Tuple[NDArray[Real], NDArray[Real]]]
    ) -> Tuple[NDArray[Real], NDArray[Real], NDArray[Real]]:
        r"""Obtain the :math:`\Sigma(z, k)` (1- or 2-D arrays) for an array of
        redshifts coming from a redshift distribution (defined by the user) and
        a single value or an array of k specified by the user. We obtain then
        :math:`\Sigma(z, k)` as a 1D array for those z and k arrays and convert
        it to a 2-D array taking into consideration the given sizes of the
        arrays for z and k. The MG parameter array goes then as a
        multiplicative factor within the MG transfer function. If k is not
        specified then only a 1D array for :math:`\Sigma(a, k=0)` is used.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        z
            A single value (e.g. for CMB) or a tuple (z, nz) describing the
            redshift distribution of the objects. Units are arbitrary because
            it is internally normalized to integrate to :math:`1`.
        """
        # Sampling scale factor from a very small (at CMB for example)
        # all the way to 1 here and today for the transfer function.
        # For a < a_single it is GR (no early MG)
        if isinstance(z, (int, float)):
            a_single = 1/(1+z)
            a = np.linspace(a_single, 1, 100)
            # a_single is for example like for the CMB surface
        else:
            if z[0] != 0.0:
                stepsize = z[1]-z[0]
                samplesize = int(z[0]/stepsize)
                z_0_to_zmin = np.linspace(0.0, z[0] - stepsize, samplesize)
                z = np.concatenate((z_0_to_zmin, z))
            a = 1 / (1 + z[::-1])
        # Scale-dependant MG case with an array of k
        lk = cosmo.get_pk_spline_lk()
        k = np.exp(lk)
        # computing MG factor array
        mgfac_1d = 1
        mgfac_1d += _Sig_MG(cosmo, k, a)
        # converting 1D MG factor to a 2D array, so it is compatible
        # with the transfer_ka input structure in MG_add.tracer and
        # add.tracer
        mgfac_2d = mgfac_1d.reshape(len(a), -1, order='F')
        # setting transfer_ka for this case
        return (a, lk, mgfac_2d)

    @warn_api
    def add_tracer(
            self,
            cosmo: Cosmology,
            *,
            kernel: Optional[Tuple[
                NDArray[Real],
                NDArray[Real]]] = None,
            transfer_ka: Optional[Tuple[
                NDArray[Real],
                NDArray[Real],
                NDArray[Real]]] = None,
            transfer_k: Optional[Tuple[
                NDArray[Real],
                NDArray[Real]]] = None,
            transfer_a: Optional[Tuple[
                NDArray[Real],
                NDArray[Real]]] = None,
            der_bessel: Literal[-1, 0, 1, 2] = 0,
            der_angles: Literal[0, 1, 2] = 0,
            is_logt: bool = False,
            extrap_order_lok: Literal[0, 1, 2] = 0,
            extrap_order_hik: Literal[0, 1, 2] = 2
    ) -> None:
        r"""Add a tracer to the collection.

        Arguments
        ---------
        cosmo
            Cosmological parameters.

        kernel
            ``(chi, w_chi)`` describing the radial kernel of the tracer. `chi`
            is the comoving radial distance (in :math:`\rm Mpc`) and
            monotonically increasing. The kernel is assumed to vanish outside
            of the input range. If not provided, a constant kernel of :math:`1`
            is assumed.

        transfer_ka
            The most general tranfer function for the tracer ``(a, lk, t_ka)``.
            Knots must be monotonically increasing. `lk` is the natural
            logarithm of wavenumber (in :math:`\rm Mpc^{-1}`). `t_ka` must have
            shape (`na, nk`). Extrapolation in `a` is continuous and constant.
            Extrapolation in `lk` is controlled by `extrap_order_xx`.

            If the transfer function is factorizable, and can be expressed as
            :math:`T(k, a) = A(a) \times K(k)`, it is more efficient to use
            `transfer_k` and `transfer_a` instead.

        transfer_k
            Scale-dependent part of a factorizable transfer function
            ``(lk, t_k)``. Knots must be monotonically increasing. `lk` is the
            natural logarithm of wavenumber (in :math:`\rm Mpc^{-1}`).
            Extrapolation is controlled by `extrap_order_xx`. If not provided,
            the :math:`k` -dependent part of the transfer function is set to
            :math:`1`. Ignored if `transfer_ka` is provided.

        transfer_a
            Time-dependent part of a factorizable transfer function
            ``(lk, t_a)``. Knots must be monotonically increasing.
            Extrapolation outside of the range is continuous and constant.
            If not provided, the :math:`a` -dependent part of the transfer
            function is set to :math:`1`. Ignored if `transfer_ka` is provided.

        der_bessel
            Order of the derivative of the Bessel functions with which this
            tracer enters the calculation of the power spectrum.

            ``-1`` for the raw functions divided by the square of the argument
            (this type of dependence is ubiquitous for many common tracers,
            e.g. lensing, IA, and makes the transfer functions more stable at
            small :math:`k` and :math:`\chi`).

        der_angles
            Flag for the the :math:`\ell`-dependent prefactor associated with
            the tracer:

            * ``0`` for no prefactor,

            * ``1`` for :math:`\ell (\ell + 1)`
              (associated withthe angular Laplacian e.g. lensing
              convergence and magnification),

            * ``2`` for :math:`\sqrt \frac{(\ell + 2)!}{(\ell - 2)!}`
              (associated with the angular derivatives of spin-2 fields,
              e.g. cosmic shear, IA).

        is_logt
            Whether `transfer_x` holds the transfer function in linear- or
            log-space.

        extrap_order_lok, extrap_order_hik
            Extrapolation order when calling the transfer function beyond the
            interpolation boundaries in :math:`k`. Extrapolated in linear- or
            log-scale, depending on `is_logt`.
        """
        is_factorizable = transfer_ka is None
        is_k_constant = (transfer_ka is None) and (transfer_k is None)
        is_a_constant = (transfer_ka is None) and (transfer_a is None)
        is_kernel_constant = kernel is None

        chi_s, wchi_s = _check_array_params(kernel, 'kernel')
        if is_factorizable:
            a_s, ta_s = _check_array_params(transfer_a, 'transfer_a')
            lk_s, tk_s = _check_array_params(transfer_k, 'transfer_k')
            tka_s = NoneArr
            if (not is_a_constant) and (a_s.shape != ta_s.shape):
                raise ValueError("Time-dependent transfer arrays "
                                 "should have the same shape")
            if (not is_k_constant) and (lk_s.shape != tk_s.shape):
                raise ValueError("Scale-dependent transfer arrays "
                                 "should have the same shape")
        else:
            a_s, lk_s, tka_s = _check_array_params(transfer_ka, 'transer_ka',
                                                   arr3=True)
            if tka_s.shape != (len(a_s), len(lk_s)):
                raise ValueError("2D transfer array has inconsistent "
                                 "shape. Should be (na,nk)")
            tka_s = tka_s.flatten()
            ta_s = NoneArr
            tk_s = NoneArr

        if not (np.diff(a_s) > 0).all():
            raise ValueError("Scale factor must be monotonically "
                             "increasing")

        status = 0
        ret = lib.cl_tracer_t_new_wrapper(cosmo.cosmo,
                                          int(der_bessel),
                                          int(der_angles),
                                          chi_s, wchi_s,
                                          a_s, lk_s,
                                          tka_s, tk_s, ta_s,
                                          int(is_logt),
                                          int(is_factorizable),
                                          int(is_k_constant),
                                          int(is_a_constant),
                                          int(is_kernel_constant),
                                          int(extrap_order_lok),
                                          int(extrap_order_hik),
                                          status)
        self._trc.append(_check_returned_tracer(ret))

    @classmethod
    def from_z_Power(
            cls,
            cosmo: Cosmology,
            *,
            A: Real,
            alpha: Real,
            z_min: Real = 0,
            z_max: Real = 6,
            n_chi: int = 1024
    ) -> Tracer:
        r"""Constructor for tracers associated with a radial kernel of the form

        .. math::

           W(\chi) = \frac{A}{(1+z)^\alpha},

        where :math:`A` is the amplitude and :math:`\alpha` is the power-law
        index. The kernel only has support in the redshift range
        :math:`z \in (z_\min, z_\max)`.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        A
            Amplitude parameter.
        alpha
            Power law index.
        z_min
            Minimum redshift of the kernel.
        z_max
            Maximum redshift of the kernel.
        n_chi
            Number of comoving radial distance intervals for kernel sampling.

        Returns
        -------

            Tracer.
        """
        if z_min >= z_max:
            raise ValueError("z_min should be smaller than z_max.")

        tracer = cls()

        chi_min = cosmo.comoving_radial_distance(1./(1+z_min))
        chi_max = cosmo.comoving_radial_distance(1./(1+z_max))
        chi_arr = np.linspace(chi_min, chi_max, n_chi)
        a_arr = cosmo.scale_factor_of_chi(chi_arr)
        w_arr = A * a_arr**alpha

        tracer.add_tracer(cosmo, kernel=(chi_arr, w_arr))
        return tracer

    def __del__(self):
        # Sometimes lib is freed before some Tracers, in which case, this
        # doesn't work.
        # So just check that lib.cl_tracer_t_free is still a real function.
        if hasattr(self, '_trc') and lib.cl_tracer_t_free is not None:
            for t in self._trc:
                lib.cl_tracer_t_free(t)


class NzTracer(Tracer):
    """Tracers with an internal redshift distribution interpolator."""

    def get_dndz(self, z):
        """Get the redshift distribution for this tracer.

        Arguments
        ---------
        z : array_like (nz,)
            Redshift.

        Returns
        -------
        array_like (nz,)
            Redshift distribution.
        """
        return self._dndz(z)


@warn_api(reorder=["has_rsd", "dndz", "bias", "mag_bias"])
def NumberCountsTracer(
        cosmo: Cosmology,
        *,
        dndz: Tuple[NDArray[Real], NDArray[Real]],
        bias: Optional[Tuple[NDArray[Real], NDArray[Real]]] = None,
        mag_bias: Optional[Tuple[NDArray[Real], NDArray[Real]]] = None,
        has_rsd: bool,
        n_samples: int = 256
) -> NzTracer:
    r"""Galaxy clustering tracer with linear scale-independent bias, including
    redshift-space distortions and magnification.

    .. note::

        For redshift-space distortions, the current implementation assumes
        linear, scale-independent growth, which is only generally true for
        :math:`\Lambda \rm CDM` and on large scales (especially when
        considering a broad :math:`N(z)`),

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    dndz
        Redshift distribution ``(z, nz)``. Units are arbitrary because it is
        internally normalized to integrate to :math:`1`.
    bias
        Galaxy bias ``(z, bz)``. If not provided, the tracer does not contain
        a term proportional to the matter density contrast.
    mag_bias
        Magnification bias ``(z, sz)``. If not provided the tracer will have no
        term associated to magnification bias.
    has_rsd
        Whether the tracer has a redshift-space distortion term.
    n_samples
        Number of eqispaced radial distance samples for the magnification
        lensing kernel. O(100) is usually enough, as the kernel is smooth.

    Returns
    -------

        Number counts tracer.
    """
    tracer = NzTracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    from scipy.interpolate import interp1d
    z_n, n = _check_array_params(dndz, 'dndz')
    with tracer.unlock():
        tracer._dndz = interp1d(z_n, n, bounds_error=False, fill_value=0)

    kernel_d = None
    if bias is not None:  # Has density term
        # Kernel
        if kernel_d is None:
            kernel_d = get_density_kernel(cosmo, dndz=dndz)
        # Transfer
        z_b, b = _check_array_params(bias, 'bias')
        # Reverse order for increasing a
        t_a = (1./(1+z_b[::-1]), b[::-1])
        tracer.add_tracer(cosmo, kernel=kernel_d, transfer_a=t_a)

    if has_rsd:  # Has RSDs
        # Kernel
        if kernel_d is None:
            kernel_d = get_density_kernel(cosmo, dndz=dndz)
        # Transfer (growth rate)
        z_b, _ = _check_array_params(dndz, 'dndz')
        a_s = 1./(1+z_b[::-1])
        t_a = (a_s, -cosmo.growth_rate(a_s))
        tracer.add_tracer(cosmo, kernel=kernel_d,
                          transfer_a=t_a, der_bessel=2)

    if mag_bias is not None:  # Has magnification bias
        # Kernel
        chi, w = get_lensing_kernel(cosmo, dndz=dndz, mag_bias=mag_bias,
                                    n_chi=n_samples)
        # Multiply by -2 for magnification
        kernel_m = (chi, -2 * w)
        if (cosmo['sigma_0'] == 0):
            # GR case
            tracer.add_tracer(cosmo, kernel=kernel_m,
                              der_bessel=-1, der_angles=1)
        else:
            # MG case
            z_b, _ = _check_array_params(dndz, 'dndz')
            tracer._MG_add_tracer(cosmo, kernel_m, z_b,
                                  der_bessel=-1, der_angles=1)
    return tracer


@warn_api
def WeakLensingTracer(
        cosmo: Cosmology,
        *,
        dndz: Tuple[NDArray[Real], NDArray[Real]],
        has_shear: bool = True,
        ia_bias: Optional[Tuple[NDArray[Real], NDArray[Real]]] = None,
        use_A_ia: bool = True,
        n_samples: int = 256
) -> NzTracer:
    r"""Tracers of galaxy shape distortions, including lensing shear and
    intrinsic alignments within the L-NLA model.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    dndz
        Redshift distribution ``(z, nz)``. Units are arbitrary because it is
        internally normalized to integrate to :math:`1`.
    has_shear
        Whether the tracer has the lensing shear contribution.
    ia_bias
        Amplite of intrinsic alignment ``(z, Aia_z)``. If not ptrovided the
        tracer will not have intrinsic alignments.
    use_A_ia
        Whether to use the conventional IA normalization. If False, use the raw
        input amplitude (which is usually :math:`1`) for use with perturbation
        theory IA modeling.
    n_samples
        Number of eqispaced radial distance samples for the magnification
        lensing kernel. O(100) is usually enough, as the kernel is smooth.

    Returns
    -------

        Weak lensing tracer.
    """
    tracer = NzTracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()

    from scipy.interpolate import interp1d
    z_n, n = _check_array_params(dndz, 'dndz')
    with tracer.unlock():
        tracer._dndz = interp1d(z_n, n, bounds_error=False, fill_value=0)

    if has_shear:
        kernel_l = get_lensing_kernel(cosmo, dndz=dndz, n_chi=n_samples)
        if (cosmo['sigma_0'] == 0):
            # GR case
            tracer.add_tracer(cosmo, kernel=kernel_l,
                              der_bessel=-1, der_angles=2)
        else:
            # MG case
            tracer._MG_add_tracer(cosmo, kernel_l, z_n,
                                  der_bessel=-1, der_angles=2)
    if ia_bias is not None:  # Has intrinsic alignments
        z_a, tmp_a = _check_array_params(ia_bias, 'ia_bias')
        # Kernel
        kernel_i = get_density_kernel(cosmo, dndz=dndz)
        if use_A_ia:
            # Normalize so that A_IA=1
            D = cosmo.growth_factor(1./(1+z_a))
            # Transfer
            # See Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
            # and note that we use C_1= 5e-14 from arXiv:0705.0166
            rho_m = physical_constants.RHO_CRITICAL * cosmo['Omega_m']
            a = - tmp_a * 5e-14 * rho_m / D
        else:
            # use the raw input normalization. Normally, this will be 1
            # to allow nonlinear PT IA models, where normalization is
            # already applied to the power spectrum.
            a = tmp_a
        # Reverse order for increasing a
        t_a = (1./(1+z_a[::-1]), a[::-1])
        tracer.add_tracer(cosmo, kernel=kernel_i, transfer_a=t_a,
                          der_bessel=-1, der_angles=2)
    return tracer


@warn_api
def CMBLensingTracer(
        cosmo: Cosmology,
        *,
        z_source: Real,
        n_samples: int = 100
) -> Tracer:
    r"""Tracer of CMB lensing :math:`\kappa`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    z_source
        Redshift of source plane for CMB lensing.
    n_samples
        Number of eqispaced radial distance samples for the magnification
        lensing kernel. O(100) is usually enough, as the kernel is smooth.

    Returns
    -------

        CMB lensing tracer.
    """
    tracer = Tracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()
    kernel = get_kappa_kernel(cosmo, z_source=z_source, n_samples=n_samples)
    if (cosmo['sigma_0'] == 0):
        tracer.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
    else:
        tracer._MG_add_tracer(cosmo, kernel, z_source,
                              der_bessel=-1, der_angles=1)
    return tracer


@warn_api
def tSZTracer(
        cosmo: Cosmology,
        *,
        z_max: Real = 6,
        n_chi: int = 1024
) -> Tracer:
    r"""Tracer of the thermal Sunyaev Zel'dovich effect (parametrized with the
    Compton-:math:`y` parameter).

    The radial kernel takes the form

    .. math::

       W(\chi) = \frac{\sigma_T}{m_ec^2} \frac{1}{1+z},

    where :math:`\sigma_T` is the Thomson scattering cross section and
    :math:`m_e` is the electron mass.

    Angular power spectra computed with this tracer, should use a 3-D power
    spectrum involving the electron pressure in physical (non-comoving) units
    of :math:`\rm eV \, cm^{-3}`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    z_max
        Maximum redshift to define the kernel.
    n_chi
        Number of intervals in comoving radial distance to sample the kernel.
    """
    # This is \sigma_T / (m_e * c^2)
    prefac = 4.01710079e-06
    return Tracer.from_z_power(cosmo, A=prefac, alpha=1, z_min=0.,
                              z_max=z_max, n_chi=n_chi)


@warn_api
def CIBTracer(
        cosmo: Cosmology,
        *,
        z_min: Real = 0,
        z_max: Real = 6,
        n_chi: int = 1024
) -> Tracer:
    r"""Tracer of the cosmic infrared background (CIB).

    The radial kernel takes the form

    .. math::

        W(\chi) = \frac{1}{1+z}.

    Angular power spectra computed with this tracer, should use a 3-D power
    spectrum involving the CIB emissivity density in units of
    :math:`\rm Jy \, Mpc^{-1} \, sr^{-1}` (or multiples thereof).

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    zmin
        Minimum redshift to define the kernel.
    z_max
        Maximum redshift to define the kernel.
    n_chi
        Number of intervals in comoving radial distance to sample the kernel.
    """
    return Tracer.from_z_power(cosmo, A=1.0, alpha=1, z_min=z_min,
                              z_max=z_max, n_chi=n_chi)


def ISWTracer(
        cosmo: Cosmology,
        *,
        z_max: Real = 6,
        n_chi: int = 1024
) -> Tracer:
    r"""Tracer of the integrated Sachs-Wolfe effect (ISW).

    This tracer is useful when cross-correlating any low-redshift probe with
    the primary CMB anisotropies. It assumes a standard Poisson equation
    relating :math:`\phi` and :math:`\delta` to linear structure growth.
    This is generally valid in :math:`\rm \Lambda CDM` cosmologies and on the
    large scales this tracer is sensitive to.

    The ISW contribution to the temperature fluctuations is:

    .. math::

        \Delta T_{\rm CMB} = 2T_{\rm CMB}
        \int_0^{\chi_{\rm CMB}}{\rm d}\chi \, a \, \dot{\phi}.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    z_max
        Maximum redshift to define the kernel.
    n_chi
        Number of intervals in comoving radial distance to sample the kernel.
    """
    tracer = Tracer()

    chi_max = cosmo.comoving_radial_distance(1./(1+z_max))
    chi = np.linspace(0, chi_max, n_chi)
    a_arr = cosmo.scale_factor_of_chi(chi)
    H0 = cosmo['h'] / physical_constants.CLIGHT_HMPC
    OM = cosmo['Omega_c'] + cosmo['Omega_b']
    Ez = cosmo.h_over_h0(a_arr)
    fz = cosmo.growth_rate(a_arr)
    w_arr = 3 * cosmo['T_CMB'] * H0**3 * OM * Ez * chi**2 * (1-fz)

    tracer.add_tracer(cosmo, kernel=(chi, w_arr), der_bessel=-1)
    return tracer


def _check_returned_tracer(return_val):
    """Wrapper to catch exceptions when tracers are spawned from C."""
    if isinstance(return_val, int):
        check(return_val)
        tr = None
    else:
        tr, _ = return_val
    return tr
