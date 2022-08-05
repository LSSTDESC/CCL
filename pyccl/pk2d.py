import warnings
import functools
import numpy as np

from . import ccllib as lib
from .errors import CCLWarning, CCLError
from .pyutils import (check, get_pk_spline_a, get_pk_spline_lk,
                      _get_spline1d_arrays, _get_spline2d_arrays)


class _Pk2D_descriptor:
    """Descriptor to enable usage of `Pk2D` class methods as instance methods.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, base):
        this = instance if instance else base

        @functools.wraps(self.func)
        def new_func(*args, **kwargs):
            return self.func(this, *args, **kwargs)
        return new_func


class Pk2D(object):
    """A power spectrum class holding the information needed to reconstruct an
    arbitrary function of wavenumber and scale factor.

    .. note::

        The ``Pk2D`` class is a wrapper around CCL's bicubic interpolators.
        Addition, subtraction, multiplication, division returning new objects
        or in-place is supported between ``Pk2D`` objects and other ``Pk2D``
        objects, integers, or floats. When the second object is also of type
        ``Pk2D``, the a- and k-range changes to the most restrictive range.
        Exponentiation is also supported for integers and floats.

    .. note::

        The power spectrum can be evaluated by directly calling the instance
        ``pk(k, a)``, or by calling the ``eval`` method, ``pk.eval(k, a)``.
        Calling the instance is vectorized in both ``k`` and ``a``.

    Parameters
    ----------
    pkfunc : :obj:`callable`
        Function with signature ``f(k, a)`` which takes vectorized input
        in ``k`` (wavenumber in :math:`\\mathrm{Mpc}^{-1}``) and a scale factor
        ``a``, and returns the value of the power spectrum, or its natural log
        depending on ``is_logp``. Power spectrum units must be compatible
        with  CCL (e.g. :math:`\\mathrm{Mpc}^3` for matter power spectrum).
        If a ``pkfunc`` is provided the power spectrum is sampled at the
        values of ``k`` and ``a`` used internally by CCL to store the linear
        and non-linear power spectra.
    a_arr : array
        An array holding values of the scale factor
    lk_arr : array
        An array holding values of the natural logarithm of the wavenumber
        (in :math:`\\mathrm{Mpc}^-1`).
    pk_arr : array
        A 2-D array of shape ``(na, nk)``, of the values of the power spectrum
        at ``a_arr`` and ``lk_arr``. Input array could be flattened, provided
        that its size is ``nk*na``. The array can hold the values of the
        natural logarithm of the power spectrum, depending on the value of
        ``is_logp``. If ``pkfunc`` is provided, all of ``a_arr``, ``lk_arr``
        and ``pk_arr`` are ignored. However, either ``pkfunc`` or all of the
        last three arrays must be provided. Note, if you pass your own Pk array
        you are responsible of making sure that it is sufficiently well-sampled
        i.e. the resolution of `a_arr` and `lk_arr` is high enough to sample
        the main features in the power spectrum. CCL uses bicubic interpolation
        to evaluate the power spectrum at any intermediate point in k and a.
    extrap_order_lok, extrap_order_hik :  {0, 1, 2}
        Extrapolation order to be used on k-values below and above the minimum
        and maximum of the splines, respectively. Note that extrapolation is
        either in log(P(k)) or in P(k), depending on the value of ``is_logp``.
    is_logp : bool
        If True, pkfunc/pkarr return/hold the natural logarithm of the power
        spectrum. Otherwise, the true value of the power spectrum is expected.
        If ``is_logp`` is ``True``, arrays are interpolate in log-space.
    cosmo : :class:`~pyccl.core.Cosmology`, optional
        Cosmological parameters.
        Used to determine sampling rates in scale factor and wavenumber.
    empty : bool
        If ``True``, just create an empty object, to be filled out later.
    """

    def __init__(self, pkfunc=None, a_arr=None, lk_arr=None, pk_arr=None,
                 is_logp=True, extrap_order_lok=1, extrap_order_hik=2,
                 cosmo=None, empty=False):
        if empty:
            return

        status = 0
        if pkfunc is None:  # Initialize power spectrum from 2D array
            # Make sure input makes sense
            if (a_arr is None) or (lk_arr is None) or (pk_arr is None):
                raise ValueError("If you do not provide a function, "
                                 "you must provide arrays")

            # Check that `a` is a monotonically increasing array.
            if not np.all((a_arr[1:] - a_arr[:-1]) > 0):
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

            # Set k and a sampling from CCL parameters
            a_arr = get_pk_spline_a(cosmo=cosmo)
            lk_arr = get_pk_spline_lk(cosmo=cosmo)

            # Compute power spectrum on 2D grid
            pkflat = np.array([pkfunc(k=np.exp(lk_arr), a=a) for a in a_arr])
            pkflat = pkflat.flatten()

        self.psp, status = lib.set_pk2d_new_from_arrays(lk_arr, a_arr, pkflat,
                                                        int(extrap_order_lok),
                                                        int(extrap_order_hik),
                                                        int(is_logp), status)
        check(status, cosmo=cosmo)

    @property
    def has_psp(self):
        return 'psp' in vars(self)

    @property
    def extrap_order_lok(self):
        return self.psp.extrap_order_lok

    @property
    def extrap_order_hik(self):
        return self.psp.extrap_order_hik

    @classmethod
    def from_model(cls, cosmo, model):
        """`Pk2D` constructor returning the power spectrum associated with
        a given numerical model.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            model (:obj:`str`): model to use. Three models allowed:
                `'bbks'` (Bardeen et al. ApJ 304 (1986) 15).
                `'eisenstein_hu'` (Eisenstein & Hu astro-ph/9709112).
                `'eisenstein_hu_nowiggles'` (Eisenstein & Hu astro-ph/9709112).
                `'emu'` (arXiv:1508.02654).
        """
        pk2d = Pk2D(empty=True)
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
            pk2d.psp, status = ret

        check(status, cosmo)
        return pk2d

    @classmethod
    @functools.wraps(from_model)
    def pk_from_model(cls, cosmo, model):
        return cls.from_model(cosmo, model)

    @_Pk2D_descriptor
    def apply_halofit(self, cosmo, pk_linear=None):
        """Pk2D constructor that applies the "HALOFIT" transformation of
        Takahashi et al. 2012 (arXiv:1208.2701) on an input linear
        power spectrum in `pk_linear`.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`):
                A Cosmology object.
            pk_linear (:class:`Pk2D`, optional):
                A :class:`Pk2D` object containing the linear power spectrum
                to transform.
        """
        if pk_linear is None:
            pk_linear = self

        if cosmo["wa"] != 0:
            # HALOFIT translates (w0, wa) to a w0_eff. This requires computing
            # the comoving distance to the CMB, which requires the background
            # splines being sampled to sufficiently high redshifts.
            cosmo.compute_distances()
            _, a = _get_spline1d_arrays(cosmo.cosmo.data.achi)
            if min(a) > 1/(1 + 3000):
                raise CCLError("Comoving distance spline does not cover "
                               "sufficiently high redshifts for HALOFIT. "
                               "HALOFIT translates (w0, wa) to a w0_eff. This "
                               "requires computing the comoving distance to "
                               "the CMB, which requires the background "
                               "splines being sampled to sufficiently high "
                               "redshifts. If using the calculator mode, "
                               "check the support of the background data.")

        pk2d = Pk2D(empty=True)
        status = 0
        ret = lib.apply_halofit(cosmo.cosmo, pk_linear.psp, status)
        if np.ndim(ret) == 0:
            status = ret
        else:
            pk2d.psp, status = ret
        check(status, cosmo)
        return pk2d

    def eval(self, k, a, cosmo=None, *, derivative=False):
        """Evaluate the power spectrum or its logarithmic derivative.

        Arguments
        ---------
        k : float or array_like
            Wavenumber value(s) in units of Mpc^-1.
        a : float
            Value of the scale factor
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmology object. Used to evaluate the power spectrum outside
            of the interpolation range in ``a``, thorugh the linear growth
            factor. If ``cosmo`` is ``None``, attempting to evaluate the power
            spectrum outside of the interpolation range will raise an error.
        derivative : bool
            If ``False``, evaluate the power spectrum. If ``True``, evaluate
            the logarithmic derivative of the power spectrum,
            :math:`\\frac{\\mathrm{d} \\log P(k)}{\\mathrm{d} \\log k}`.

        Returns
        -------
        P(k, a) : float or array_like
            Value(s) of the power spectrum.
        """
        # determine if logarithmic derivative is needed
        if not derivative:
            eval_funcs = lib.pk2d_eval_single, lib.pk2d_eval_multi
        else:
            eval_funcs = lib.pk2d_der_eval_single, lib.pk2d_der_eval_multi

        # handle scale factor extrapolation
        if cosmo is None:
            cosmo = self.eval._cosmo
            self.psp.extrap_linear_growth = 404  # flag no extrapolation
        else:
            cosmo.compute_growth()  # growth factors for extrapolation
            self.psp.extrap_linear_growth = 401  # flag extrapolation

        status = 0
        if isinstance(k, int):
            k = float(k)
        if isinstance(k, float):
            f, status = eval_funcs[0](self.psp, np.log(k), a,
                                      cosmo.cosmo, status)
        else:
            k_use = np.atleast_1d(k)
            f, status = eval_funcs[1](self.psp, np.log(k_use), a,
                                      cosmo.cosmo, k_use.size, status)

        # Catch scale factor extrapolation bounds error.
        if status == lib.CCL_ERROR_SPLINE_EV:
            raise TypeError(
                "Pk2D evaluation scale factor is outside of the "
                "interpolation range. To extrapolate, pass a Cosmology.")
        check(status, cosmo)
        return f

    # Save a dummy cosmology as an attribute of the `eval` method so we don't
    # have to initialize one every time no `cosmo` is passed. This is gentle
    # with memory too, as `free` does not work for an empty cosmology.
    eval._cosmo = type("Dummy", (object,), {"cosmo": lib.cosmology()})()

    def eval_dlogpk_dlogk(self, k, a, cosmo):
        """Evaluate logarithmic derivative. See ``Pk2D.eval`` for details."""
        return self.eval(k, a, cosmo=cosmo, derivative=True)

    def __call__(self, k, a, cosmo=None, *, derivative=False):
        """Callable vectorized instance."""
        out = np.array([self.eval(k, aa, cosmo=cosmo, derivative=derivative)
                        for aa in np.atleast_1d(a).astype(float)])
        return out.squeeze()[()]

    def copy(self):
        """Return a copy of this Pk2D object."""
        if not self:
            return Pk2D(empty=True)
        return self + 0

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
        if not self:
            raise ValueError("Pk2D object does not have data.")

        a_arr, lk_arr, pk_arr = _get_spline2d_arrays(self.psp.fka)
        if self.psp.is_log:
            pk_arr = np.exp(pk_arr)

        return a_arr, lk_arr, pk_arr

    def __del__(self):
        """Free memory associated with this Pk2D structure."""
        if self:
            lib.f2d_t_free(self.psp)

    def __bool__(self):
        return self.has_psp

    def __contains__(self, other):
        if not (self.psp.lkmin <= other.psp.lkmin
                and self.psp.lkmax >= other.psp.lkmax
                and self.psp.amin <= other.psp.amin
                and self.psp.amax >= other.psp.amax):
            return False
        return True

    def _get_binary_operator_arrays(self, other):
        if not (self and other):
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

    def __sub__(self, other):
        return self + (-1)*other

    def __truediv__(self, other):
        return self * other**(-1)

    __radd__ = __add__

    __rmul__ = __mul__

    def __rsub__(self, other):
        return other + (-1)*self

    def __rtruediv__(self, other):
        return other * self**(-1)

    def __iadd__(self, other):
        self = self + other
        return self

    def __imul__(self, other):
        self = self * other
        return self

    def __isub__(self, other):
        self = self - other
        return self

    def __itruediv__(self, other):
        self = self / other
        return self

    def __ipow__(self, other):
        self = self**other
        return self


def parse_pk2d(cosmo, p_of_k_a, is_linear=False):
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
