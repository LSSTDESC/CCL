import warnings
import numpy as np

from . import ccllib as lib
from .errors import CCLWarning, CCLError
from .pyutils import (check, get_pk_spline_a, get_pk_spline_lk,
                      _get_spline1d_arrays, _get_spline2d_arrays)


class Pk2D(object):
    """A power spectrum class holding the information needed to reconstruct an
    arbitrary function of wavenumber and scale factor.

    Args:
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
        extrap_order_lok (int): extrapolation order to be used on k-values
             below the minimum of the splines (use 0, 1 or 2). Note that
             the extrapolation will be done in either log(P(k)) or P(k),
             depending on the value of `is_logp`.
        extrap_order_hik (int): extrapolation order to be used on k-values
             above the maximum of the splines (use 0, 1 or 2). Note that
             the extrapolation will be done in either log(P(k)) or P(k),
             depending on the value of `is_logp`.
        is_logp (boolean): if True, pkfunc/pkarr return/hold the natural
             logarithm of the power spectrum. Otherwise, the true value
             of the power spectrum is expected. Note that arrays will be
             interpolated in log space if `is_logp` is set to `True`.
        cosmo (:class:`~pyccl.core.Cosmology`, optional): Cosmology object.
             Used to determine sampling rates in scale factor and wavenumber.
        empty (bool): if True, just create an empty object, to be filled
            out later
    """
    def __init__(self, pkfunc=None, a_arr=None, lk_arr=None, pk_arr=None,
                 is_logp=True, extrap_order_lok=1, extrap_order_hik=2,
                 cosmo=None, empty=False):
        if empty:
            self.has_psp = False
            return

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

        self.extrap_order_lok = extrap_order_lok
        self.extrap_order_hik = extrap_order_hik

        status = 0
        self.psp, status = lib.set_pk2d_new_from_arrays(lk_arr, a_arr, pkflat,
                                                        int(extrap_order_lok),
                                                        int(extrap_order_hik),
                                                        int(is_logp), status)
        check(status, cosmo=cosmo)
        self.has_psp = True

    @classmethod
    def pk_from_model(Pk2D, cosmo, model):
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
        pk2d.has_psp = True
        return pk2d

    @classmethod
    def apply_halofit(Pk2D, cosmo, pk_linear):
        """Pk2D constructor that applies the "HALOFIT" transformation of
        Takahashi et al. 2012 (arXiv:1208.2701) on an input linear
        power spectrum in `pk_linear`.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            pk_linear (:class:`Pk2D`): a :class:`Pk2D` object containing
                the linear power spectrum to transform.
        """
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
        pk2d.has_psp = True
        return pk2d

    def eval(self, k, a, cosmo):
        """Evaluate power spectrum.

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
                cosmology is needed).

        Returns:
            float or array_like: value(s) of the power spectrum.
        """
        # make sure we have growth factors for extrapolation
        cosmo.compute_growth()

        status = 0
        cospass = cosmo.cosmo

        if isinstance(k, int):
            k = float(k)
        if isinstance(k, float):
            f, status = lib.pk2d_eval_single(self.psp, np.log(k), a, cospass,
                                             status)
        else:
            k_use = np.atleast_1d(k)
            f, status = lib.pk2d_eval_multi(self.psp, np.log(k_use),
                                            a, cospass,
                                            k_use.size, status)
        check(status, cosmo)

        return f

    def eval_dlogpk_dlogk(self, k, a, cosmo):
        """Evaluate logarithmic derivative

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
                cosmology is needed).

        Returns:
            float or array_like: value(s) of the power spectrum.
        """
        # make sure we have growth factors for extrapolation
        cosmo.compute_growth()

        status = 0
        cospass = cosmo.cosmo

        if isinstance(k, int):
            k = float(k)
        if isinstance(k, float):
            f, status = lib.pk2d_der_eval_single(self.psp, np.log(k), a,
                                                 cospass, status)
        else:
            k_use = np.atleast_1d(k)
            f, status = lib.pk2d_der_eval_multi(self.psp, np.log(k_use),
                                                a, cospass,
                                                k_use.size, status)
        check(status, cosmo)

        return f

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
        """Free memory associated with this Pk2D structure
        """
        if hasattr(self, 'has_psp'):
            if self.has_psp and hasattr(self, 'psp'):
                lib.f2d_t_free(self.psp)

    def _get_binary_operator_arrays(self, other):
        if not isinstance(other, Pk2D):
            raise TypeError("Binary operator of Pk2D objects is only defined "
                            "for other Pk2D objects.")
        if not (self.has_psp and other.has_psp):
            raise ValueError("Pk2D object does not have data.")
        if (self.psp.lkmin < other.psp.lkmin
                or self.psp.lkmax > other.psp.lkmax):
            raise ValueError("The 2nd operand has its data defined over a "
                             "smaller k range than the 1st operand. To avoid "
                             "extrapolation, this operation is forbidden. If "
                             "you want to operate on the smaller support, "
                             "try swapping the operands.")
        if (self.psp.amin < other.psp.amin
                or self.psp.amax > other.psp.amax):
            raise ValueError("The 2nd operand has its data defined over a "
                             "smaller a range than the 1st operand. To avoid "
                             "extrapolation, this operation is forbidden. If "
                             "you want to operate on the smaller support, "
                             "try swapping the operands.")

        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        a_arr_b, lk_arr_b, pk_arr_b = other.get_spline_arrays()
        if not (a_arr_a.size == a_arr_b.size and lk_arr_a.size == lk_arr_b.size
                and np.allclose(a_arr_a, a_arr_b)
                and np.allclose(lk_arr_a, lk_arr_b)):
            warnings.warn("The arrays of the two Pk2D objects are defined at "
                          "different points in k and/or a. The second operand "
                          "will be interpolated for the operation.\n"
                          "The resulting Pk2D object will be defined for "
                          f"{self.psp.lkmin} <= log k <= {self.psp.lkmax} and "
                          f"{self.psp.amin} <= a <= {self.psp.amax}.",
                          category=CCLWarning)

            # Since the power spectrum is evalulated on a smaller support than
            # where it was defined, no extrapolation is necessary and the
            # dependence on the cosmology in moot.
            # CosmologyVanillaLCDM is being imported here instead of the top of
            # the module due to circular import issues there.
            from .core import CosmologyVanillaLCDM
            dummy_cosmo = CosmologyVanillaLCDM()
            pk_arr_b = np.array([other.eval(k=np.exp(lk_arr_a),
                                            a=a_,
                                            cosmo=dummy_cosmo)
                                 for a_ in a_arr_a])
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

    def __radd__(self, other):
        return self.__add__(other)

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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        """Take a Pk2D instance to a power.
        """
        if not isinstance(exponent, (float, int)):
            raise TypeError("Exponentiation of Pk2D is only defined for "
                            "floats and ints.")
        a_arr_a, lk_arr_a, pk_arr_a = self.get_spline_arrays()
        if np.any(pk_arr_a < 0) and exponent % 1 != 0:
            warnings.warn("Taking a non-positive Pk2D object to a non-integer "
                          "power may lead to unexpected results",
                          category=CCLWarning)

        pk_arr_new = pk_arr_a**exponent

        logp = np.all(pk_arr_new > 0)
        if logp:
            pk_arr_new = np.log(pk_arr_new)

        new = Pk2D(a_arr=a_arr_a, lk_arr=lk_arr_a, pk_arr=pk_arr_new,
                   is_logp=logp,
                   extrap_order_lok=self.extrap_order_lok,
                   extrap_order_hik=self.extrap_order_hik)

        return new


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
