import warnings
import functools
import numpy as np

from . import ccllib as lib
from .base import (CCLObject, UnlockInstance, unlock_instance,
                   warn_api, deprecated)
from ._repr import _build_string_Pk2D
from .errors import CCLWarning, CCLError, CCLDeprecationWarning
from .pyutils import check, _get_spline2d_arrays


class _Pk2D_descriptor:
    """Descriptor to deprecate usage of `Pk2D` methods as class methods."""
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, base):
        if instance is None:
            warnings.warn("Use of the power spectrum as an argument "
                          f"is deprecated in {self.func.__name__}. "
                          "Use the instance method instead.",
                          CCLDeprecationWarning)
            this = base
        else:
            this = instance

        @functools.wraps(self.func)
        def new_func(*args, **kwargs):
            return self.func(this, *args, **kwargs)

        return new_func


class Pk2D(CCLObject):
    """A power spectrum class holding the information needed to reconstruct an
    arbitrary function of wavenumber and scale factor.

    Args:
        a_arr (array): an array holding values of the scale factor
        lk_arr (array): an array holding values of the natural logarithm
             of the wavenumber (in units of Mpc^-1).
        pk_arr (array): a 2D array containing the values of the power
             spectrum at the values of the scale factor and the wavenumber
             held by `a_arr` and `lk_arr`. The shape of this array must be
             `[na,nk]`, where `na` is the size of `a_arr` and `nk` is the
             size of `lk_arr`. This array can be provided in a flattened
             form as long as the total size matches `nk*na`. The array can
             hold the values of the natural logarithm of the power
             spectrum, depending on the value of `is_logp`. If `pkfunc`
             is not None, then `a_arr`, `lk_arr` and `pk_arr` are ignored.
             However, either `pkfunc` or all of the last three array must
             be non-None. Note that, if you pass your own Pk array, you
             are responsible of making sure that it is sufficiently well
             sampled (i.e. the resolution of `a_arr` and `lk_arr` is high
             enough to sample the main features in the power spectrum).
             For reference, CCL will use bicubic interpolation to evaluate
             the power spectrum at any intermediate point in k and a.
        pkfunc (:obj:`function`): a function returning a floating point
             number or numpy array with the signature `f(k,a)`, where k
             is a wavenumber (in units of Mpc^-1) and a is the scale
             factor. The function must able to take numpy arrays as `k`.
             The function must return the value(s) of the power spectrum
             (or its natural logarithm, depending on the value of
             `is_logp`. The power spectrum units should be compatible
             with those used by CCL (e.g. if you're passing a matter power
             spectrum, its units should be Mpc^3). If this argument is not
             `None`, this function will be sampled at the values of k and
             a used internally by CCL to store the linear and non-linear
             power spectra.
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object. The cosmology
             object is needed in order if `pkfunc` is not `None`. The object is
             used to determine the sampling rate in scale factor and
             wavenumber.
        is_logp (boolean): if True, pkfunc/pkarr return/hold the natural
             logarithm of the power spectrum. Otherwise, the true value
             of the power spectrum is expected. Note that arrays will be
             interpolated in log space if `is_logp` is set to `True`.
        extrap_order_lok (int): extrapolation order to be used on k-values
             below the minimum of the splines (use 0, 1 or 2). Note that
             the extrapolation will be done in either log(P(k)) or P(k),
             depending on the value of `is_logp`.
        extrap_order_hik (int): extrapolation order to be used on k-values
             above the maximum of the splines (use 0, 1 or 2). Note that
             the extrapolation will be done in either log(P(k)) or P(k),
             depending on the value of `is_logp`.
        empty (bool): if True, just create an empty object, to be filled
            out later
    """
    __repr__ = _build_string_Pk2D

    @warn_api(reorder=["pkfunc", "a_arr", "lk_arr", "pk_arr", "is_logp",
                       "extrap_order_lok", "extrap_order_hik", "cosmo"])
    def __init__(self, *, a_arr=None, lk_arr=None, pk_arr=None,
                 pkfunc=None, cosmo=None, is_logp=True,
                 extrap_order_lok=1, extrap_order_hik=2,
                 empty=False):
        # set extrapolation order before everything else
        # in case an empty Pk2D is created
        self.extrap_order_lok = extrap_order_lok
        self.extrap_order_hik = extrap_order_hik

        if empty:
            self.has_psp = False
            return

        status = 0
        if pkfunc is None:  # Initialize power spectrum from 2D array
            # Make sure input makes sense
            if (a_arr is None) or (lk_arr is None) or (pk_arr is None):
                raise ValueError("If you do not provide a function, "
                                 "you must provide arrays")

            # Check that `a` is a monotonically increasing array.
            if not np.array_equal(a_arr, np.sort(a_arr)):
                raise ValueError("Input scale factor array in `a_arr` is not "
                                 "monotonically increasing.")

            pkflat = pk_arr.flatten()
            # Check dimensions make sense
            if (len(a_arr)*len(lk_arr) != len(pkflat)):
                raise ValueError("Size of input arrays is inconsistent")
        else:  # Initialize power spectrum from function
            # Check that the input function has the right signature
            try:
                pkfunc(k=np.array([1E-2, 2E-2]), a=0.5)
            except Exception:
                raise ValueError("Can't use input function")

            if cosmo is None:
                raise ValueError("A cosmology is needed if initializing "
                                 "power spectrum from a function")

            # Set k and a sampling from CCL parameters
            nk = lib.get_pk_spline_nk(cosmo.cosmo)
            na = lib.get_pk_spline_na(cosmo.cosmo)
            a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
            check(status)
            lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
            check(status)

            # Compute power spectrum on 2D grid
            pkflat = np.zeros([na, nk])
            for ia, a in enumerate(a_arr):
                pkflat[ia, :] = pkfunc(k=np.exp(lk_arr), a=a)
            pkflat = pkflat.flatten()

        self.psp, status = lib.set_pk2d_new_from_arrays(lk_arr, a_arr, pkflat,
                                                        int(extrap_order_lok),
                                                        int(extrap_order_hik),
                                                        int(is_logp), status)
        check(status)
        self.has_psp = True

    @classmethod
    def from_model(cls, cosmo, model):
        """`Pk2D` constructor returning the power spectrum associated with
        a given numerical model.

        Arguments:
            cosmo (:class:`~pyccl.core.Cosmology`)
                A Cosmology object.
            model (:obj:`str`)
                model to use. These models allowed:
                  - `'bbks'` (Bardeen et al. ApJ 304 (1986) 15)
                  - `'eisenstein_hu'` (Eisenstein & Hu astro-ph/9709112)
                  - `'eisenstein_hu_nowiggles'` (Eisenstein & Hu astro-ph/9709112)
                  - `'emu'` (arXiv:1508.02654).

        Returns:
            :class:`~pyccl.pk2d.Pk2D`
                The power spectrum of the input model.
        """  # noqa
        if model in ['bacco', ]:  # other emulators go in here
            from .emulator import PowerSpectrumEmulator as PSE
            return PSE.from_name(model)().get_pk_linear(cosmo)

        pk2d = cls(empty=True)
        status = 0
        if model == 'bbks':
            cosmo.compute_growth()
            ret = lib.compute_linpower_bbks(cosmo.cosmo, status)
        elif model == 'eisenstein_hu':
            cosmo.compute_growth()
            ret = lib.compute_linpower_eh(cosmo.cosmo, 1, status)
        elif model == 'eisenstein_hu_nowiggles':
            cosmo.compute_growth()
            ret = lib.compute_linpower_eh(cosmo.cosmo, 0, status)
        elif model == 'emu':
            ret = lib.compute_power_emu(cosmo.cosmo, status)
        else:
            raise ValueError("Unknown model %s " % model)

        if np.ndim(ret) == 0:
            status = ret
        else:
            with UnlockInstance(pk2d):
                pk2d.psp, status = ret

        check(status, cosmo)
        with UnlockInstance(pk2d):
            pk2d.has_psp = True
        return pk2d

    @classmethod
    @functools.wraps(from_model)
    @deprecated(new_function=from_model.__func__)
    def pk_from_model(cls, cosmo, model):
        return cls.from_model(cosmo, model)

    @_Pk2D_descriptor
    def apply_halofit(self, cosmo, *, pk_linear=None):
        """Pk2D constructor that applies the "HALOFIT" transformation of
        Takahashi et al. 2012 (arXiv:1208.2701) on an input linear power
        spectrum in `pk_linear`. See ``Pk2D.apply_nonlin_model`` for details.
        """
        if pk_linear is not None:
            self = pk_linear

        pk2d = self.__class__(empty=True)
        status = 0
        ret = lib.apply_halofit(cosmo.cosmo, self.psp, status)
        if np.ndim(ret) == 0:
            status = ret
        else:
            with UnlockInstance(pk2d):
                pk2d.psp, status = ret
        check(status, cosmo)
        with UnlockInstance(pk2d):
            pk2d.has_psp = True
        return pk2d

    @_Pk2D_descriptor
    def apply_nonlin_model(self, cosmo, model, *, pk_linear=None):
        """Pk2D constructor that applies a non-linear model
        to a linear power spectrum.

        Arguments:
            cosmo (:class:`~pyccl.core.Cosmology`)
                A Cosmology object.
            model (str)
                Model to use.
            pk_linear (:class:`Pk2D`)
                A :class:`Pk2D` object containing the linear power spectrum
                to transform. This argument is deprecated and will be removed
                in a future release. Use the instance method instead.

        Returns:
            :class:`Pk2D`:
                The transormed power spectrum.
        """
        if pk_linear is not None:
            self = pk_linear

        if model == "halofit":
            pk2d_new = self.apply_halofit(cosmo)
        elif model in ["bacco", ]:  # other emulator names go in here
            from .emulator import PowerSpectrumEmulator as PSE
            emu = PSE.from_name(model)()
            pk2d_new = emu.apply_nonlin_model(cosmo, self)
        return pk2d_new

    @_Pk2D_descriptor
    def include_baryons(self, cosmo, model, *, pk_nonlin=None):
        """Pk2D constructor that applies a correction for baryons to
        a non-linear power spectrum.

        Arguments:
            cosmo (:class:`~pyccl.core.Cosmology`):
                A Cosmology object.
            model (str):
                Model to use.
            pk_nonlin (:class:`Pk2D`):
                A :class:`Pk2D` object containing the non-linear power
                spectrum to transform. This argument is deprecated and will be
                removed in a future release. Use the instance method instead.

        Returns:
            :class:`Pk2D`
                A copy of the input power spectrum where the nonlinear model
                has been applied.
        """
        if pk_nonlin is not None:
            self = pk_nonlin
        return cosmo.baryon_correct(model, self)

    def eval(self, k, a, cosmo=None, *, derivative=False):
        """Evaluate power spectrum or its logarithmic derivative:

        .. math::
           \\frac{d\\log P(k,a)}{d\\log k}

        Args:
            k (float or array_like): wavenumber value(s) in units of Mpc^-1.
            a (float): value of the scale factor
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object. The
                cosmology object is needed in order to evaluate the power
                spectrum outside the interpolation range in `a`. E.g. if you
                want to evaluate the power spectrum at a very small a, not
                covered by the arrays you passed when initializing this object,
                the power spectrum will be extrapolated from the earliest
                available value using the linear growth factor (for which a
                cosmology is needed). If no Cosmology is passed, attempting
                to evaluate the power spectrum outside of the scale factor
                boundaries will raise an exception.

        Returns:
            float or array_like: value(s) of the power spectrum.
        """
        # determine if logarithmic derivative is needed
        if not derivative:
            eval_funcs = lib.pk2d_eval_single, lib.pk2d_eval_multi
        else:
            eval_funcs = lib.pk2d_der_eval_single, lib.pk2d_der_eval_multi

        # handle scale factor extrapolation
        if cosmo is None:
            from .core import CosmologyVanillaLCDM
            cosmo_use = CosmologyVanillaLCDM()  # this is not used anywhere
            self.psp.extrap_linear_growth = 404  # flag no extrapolation
        else:
            cosmo_use = cosmo
            # make sure we have growth factors for extrapolation
            cosmo.compute_growth()

        status = 0
        cospass = cosmo_use.cosmo

        if isinstance(k, int):
            k = float(k)
        if isinstance(k, float):
            f, status = eval_funcs[0](self.psp, np.log(k), a, cospass, status)
        else:
            k_use = np.atleast_1d(k)
            f, status = eval_funcs[1](self.psp, np.log(k_use), a, cospass,
                                      k_use.size, status)

        # handle scale factor extrapolation
        if cosmo is None:
            self.psp.extrap_linear_growth = 401  # revert flag 404

        try:
            check(status, cosmo_use)
        except CCLError as err:
            if (cosmo is None) and ("CCL_ERROR_SPLINE_EV" in str(err)):
                raise TypeError(
                    "Pk2D evaluation scale factor is outside of the "
                    "interpolation range. To extrapolate, pass a "
                    "Cosmology.", err)
            else:
                raise err

        return f

    def eval_dlPk_dlk(self, k, a, cosmo=None):
        """Evaluate logarithmic derivative. See ``Pk2d.eval`` for details."""
        f = self.eval(k, a, cosmo=cosmo, derivative=True)
        return f

    @functools.wraps(eval_dlPk_dlk)
    @deprecated(eval_dlPk_dlk)
    def eval_dlogpk_dlogk(self, k, a, cosmo):
        return self.eval_dlPk_dlk(k, a, cosmo)

    def __call__(self, k, a, cosmo=None, *, derivative=False):
        """Callable vectorized instance."""
        out = np.array([self.eval(k, aa, cosmo, derivative=derivative)
                        for aa in np.atleast_1d(a).astype(float)])
        return out.squeeze()[()]

    def copy(self):
        """Return a copy of this Pk2D object."""
        if not self.has_psp:
            return Pk2D(empty=True)

        a_arr, lk_arr, pk_arr = self.get_spline_arrays()

        is_logp = bool(self.psp.is_log)
        if is_logp:
            # log in-place
            np.log(pk_arr, out=pk_arr)

        pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=is_logp,
                    extrap_order_lok=self.psp.extrap_order_lok,
                    extrap_order_hik=self.psp.extrap_order_hik)

        return pk2d

    def get_spline_arrays(self):
        """Get the spline data arrays.

        Returns:
            a_arr: array_like
                Array of scale factors.
            lk_arr: array_like
                Array of logarithm of wavenumber k.
            pk_arr: array_like
                Array of the power spectrum P(k, z). The shape
                is (a_arr.size, lk_arr.size).
        """
        if not self.has_psp:
            raise ValueError("Pk2D object does not have data.")

        a_arr, lk_arr, pk_arr = _get_spline2d_arrays(self.psp.fka)
        if self.psp.is_log:
            pk_arr = np.exp(pk_arr)

        return a_arr, lk_arr, pk_arr

    def __del__(self):
        """Free memory associated with this Pk2D structure."""
        if hasattr(self, 'has_psp'):
            if self.has_psp and hasattr(self, 'psp'):
                lib.f2d_t_free(self.psp)

    def __contains__(self, other):
        if not (self.psp.lkmin <= other.psp.lkmin
                and self.psp.lkmax >= other.psp.lkmax
                and self.psp.amin <= other.psp.amin
                and self.psp.amax >= other.psp.amax):
            return False
        return True

    def _get_binary_operator_arrays(self, other):
        if not (self.has_psp and other.has_psp):
            raise ValueError("Pk2D object does not have data.")
        if self not in other:
            raise ValueError(
                "The 2nd operand has its data defined over a smaller range "
                "than the 1st operand. To avoid extrapolation, this operation "
                "is forbidden. If you want to operate on the smaller support, "
                "try swapping the operands.")

        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        a_arr_b, lk_arr_b, pk_arr_b = other.get_spline_arrays()
        if not (a_arr_a.size == a_arr_b.size
                and lk_arr_a.size == lk_arr_b.size
                and np.allclose(a_arr_a, a_arr_b)
                and np.allclose(lk_arr_a, lk_arr_b)):
            warnings.warn(
                "Operands defined over different ranges. "
                "The result will be interpolated and clipped to "
                f"{self.psp.lkmin} <= log k <= {self.psp.lkmax} and "
                f"{self.psp.amin} <= a <= {self.psp.amax}.", CCLWarning)
            pk_arr_b = other(np.exp(lk_arr_a), a_arr_a)

        return a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b

    def __add__(self, other):
        """Adds two Pk2D instances.

        The a and k ranges of the 2nd operand need to be the same or smaller
        than the 1st operand.
        The returned Pk2D object uses the same a and k arrays as the first
        operand.
        """
        if isinstance(other, (float, int)):
            a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
            pk_arr_new = pk_arr_a + other
        elif isinstance(other, Pk2D):
            a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b = \
                self._get_binary_operator_arrays(other)
            pk_arr_new = pk_arr_a + pk_arr_b
        else:
            raise TypeError("Addition of Pk2D is only defined for "
                            "floats, ints, and Pk2D objects.")

        logp = np.all(pk_arr_new > 0)
        if logp:
            pk_arr_new = np.log(pk_arr_new)

        new = Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                   is_logp=logp,
                   extrap_order_lok=self.extrap_order_lok,
                   extrap_order_hik=self.extrap_order_hik)

        return new

    __radd__ = __add__

    @unlock_instance(mutate=True)
    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def __sub__(self, other):
        return self + (-1)*other

    __rsub__ = __sub__

    @unlock_instance(mutate=True)
    def __isub__(self, other):
        self = self.__sub__(other)
        return self

    def __mul__(self, other):
        """Multiply two Pk2D instances.

        The a and k ranges of the 2nd operand need to be the same or smaller
        than the 1st operand.
        The returned Pk2D object uses the same a and k arrays as the first
        operand.
        """
        if isinstance(other, (float, int)):
            a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
            pk_arr_new = other * pk_arr_a
        elif isinstance(other, Pk2D):
            a_arr_a, lk_arr_a, pk_arr_a, pk_arr_b = \
                self._get_binary_operator_arrays(other)
            pk_arr_new = pk_arr_a * pk_arr_b
        else:
            raise TypeError("Multiplication of Pk2D is only defined for "
                            "floats, ints, and Pk2D objects.")

        logp = np.all(pk_arr_new > 0)
        if logp:
            pk_arr_new = np.log(pk_arr_new)

        new = Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                   is_logp=logp,
                   extrap_order_lok=self.extrap_order_lok,
                   extrap_order_hik=self.extrap_order_hik)
        return new

    __rmul__ = __mul__

    @unlock_instance(mutate=True)
    def __imul__(self, other):
        self = self.__mul__(other)
        return self

    def __truediv__(self, other):
        return self * other**(-1)

    __rtruediv__ = __truediv__

    @unlock_instance(mutate=True)
    def __itruediv__(self, other):
        self = self.__div__(other)
        return self

    def __pow__(self, exponent):
        """Take a Pk2D instance to a power.
        """
        if not isinstance(exponent, (float, int)):
            raise TypeError(
                "Exponentiation of Pk2D is only defined for floats and ints.")
        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        if np.any(pk_arr_a < 0) and exponent % 1 != 0:
            warnings.warn(
                "Taking a non-positive Pk2D object to a non-integer "
                "power may lead to unexpected results", CCLWarning)

        pk_arr_new = pk_arr_a**exponent

        logp = np.all(pk_arr_new > 0)
        if logp:
            pk_arr_new = np.log(pk_arr_new)

        new = Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                   is_logp=logp,
                   extrap_order_lok=self.extrap_order_lok,
                   extrap_order_hik=self.extrap_order_hik)

        return new

    __rpow__ = __pow__

    @unlock_instance(mutate=True)
    def __ipow__(self, other):
        self = self.__pow__(other)
        return self


@warn_api
def parse_pk2d(cosmo, p_of_k_a, *, is_linear=False):
    """ Return the C-level `f2d` spline associated with a
    :class:`Pk2D` object.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        p_of_k_a (:class:`Pk2D`, :obj:`str` or `None`): if a
            :class:`Pk2D` object, its `f2d` spline will be used. If
            a string, the linear or non-linear power spectrum stored
            by `cosmo` under this name will be used. If `None`, the
            matter power spectrum stored by `cosmo` will be used.
        is_linear (:obj:`bool`): if `True`, and if `p_of_k_a` is a
            string or `None`, the linear version of the corresponding
            power spectrum will be used (otherwise it'll be the
            non-linear version).
    """
    if isinstance(p_of_k_a, Pk2D):
        psp = p_of_k_a.psp
    else:
        if (p_of_k_a is None) or isinstance(p_of_k_a, str):
            name = p_of_k_a
        else:
            raise ValueError("p_of_k_a must be a pyccl.Pk2D object, "
                             "a string, or None")

        if is_linear:
            cosmo.compute_linear_power()
            pk = cosmo.get_linear_power(name)
        else:
            cosmo.compute_nonlin_power()
            pk = cosmo.get_nonlin_power(name)
        psp = pk.psp
    return psp
