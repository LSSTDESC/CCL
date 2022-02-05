import sys
import warnings
import numpy as np

from . import ccllib as lib

from .errors import CCLWarning, CCLError
from .pyutils import check, _get_spline2d_arrays, warn_api, deprecated
from ._pk2d import (
    _Pk2D_descriptor, from_model, pk_from_model, apply_halofit,
    apply_nonlin_model, include_baryons)


class Pk2D(object):
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
    from_model = classmethod(from_model)
    pk_from_model = classmethod(pk_from_model)
    apply_halofit = _Pk2D_descriptor(apply_halofit)
    apply_nonlin_model = _Pk2D_descriptor(apply_nonlin_model)
    include_baryons = _Pk2D_descriptor(include_baryons)

    @warn_api(order=["pkfunc", "a_arr", "lk_arr", "pk_arr", "is_logp",
                     "extrap_order_lok", "extrap_order_hik", "cosmo"])
    def __init__(self, *, a_arr=None, lk_arr=None, pk_arr=None,
                 pkfunc=None, cosmo=None, is_logp=True,
                 extrap_order_lok=1, extrap_order_hik=2,
                 empty=False):
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

        self.extrap_order_lok = extrap_order_lok
        self.extrap_order_hik = extrap_order_hik

        self.psp, status = lib.set_pk2d_new_from_arrays(lk_arr, a_arr, pkflat,
                                                        int(extrap_order_lok),
                                                        int(extrap_order_hik),
                                                        int(is_logp), status)
        check(status)
        self.has_psp = True

    def eval(self, k, a, cosmo=None, *, derivative=False):
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
            self.psp.extrap_linear_growth = 401  # revert flag linear growth

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
                cosmology is needed). If no Cosmology is passed, attempting
                to evaluate the power spectrum outside of the scale factor
                boundaries will raise an exception.

        Returns:
            float or array_like: value(s) of the power spectrum.
        """
        f = self.eval(k, a, cosmo=cosmo, derivative=True)
        return f

    @deprecated(eval_dlPk_dlk)
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
                cosmology is needed). If no Cosmology is passed, attempting
                to evaluate the power spectrum outside of the scale factor
                boundaries will raise an exception.

        Returns:
            float or array_like: value(s) of the power spectrum.
        """
        return self.eval_dlPk_dlk(k, a, cosmo)

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

    def copy(self):
        """Return a copy of this Pk2D object."""
        if not self.has_psp:
            pk2d = Pk2D(extrap_order_lok=self.extrap_order_lok,
                        extrap_order_hik=self.extrap_order_hik,
                        empty=True)
            return pk2d

        a_arr, lk_arr, pk_arr = _get_spline2d_arrays(self.psp.fka)
        pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=self.psp.is_log,
                    extrap_order_lok=self.psp.extrap_order_lok,
                    extrap_order_hik=self.psp.extrap_order_hik)

        return pk2d

    def __del__(self):
        """Free memory associated with this Pk2D structure
        """
        if hasattr(self, 'has_psp'):
            if self.has_psp and hasattr(self, 'psp'):
                lib.f2d_t_free(self.psp)

    def __eq__(self, other):
        """Check if two Pk2D objects are equivalent, i.e. they contain the
        same data over the same range.
        """
        return hash(self) == hash(other)

    def __hash__(self):
        """Compute the hash of this ``Pk2D`` object."""
        return hash(repr(self)) + sys.maxsize + 1

    def __repr__(self):
        """Construct a string for this ``Pk2D`` object.
        If this object has data, the data arrays are replaced by their hash.
        """
        s = "pyccl.Pk2D\n"
        s += f"  extrap_lok  =  {self.extrap_order_lok}\n"
        s += f"  extrap_hik  =  {self.extrap_order_hik}\n"
        if self.has_psp:
            # print the first and last elements of the arrays
            # also print the hashes of the arrays
            a_arr, lk_arr, pk_arr = self.get_spline_arrays()
            H = [hash(arr.tobytes()) + sys.maxsize + 1
                 for arr in [a_arr, lk_arr, pk_arr]]
            s += f"  a_arr   =  {a_arr.min():6.1f} .. {a_arr.max():6.1f}"
            s += f"    #{H[0]:20d}\n"
            s += f"  lk_arr  =  {lk_arr.min():6.3f} .. {lk_arr.max():6.3f}"
            s += f"    #{H[1]:20d}\n"
            s += f"  pk_arr  =  {pk_arr[0, 0]:6.3f} .. {pk_arr[-1, -1]:6.3f}"
            s += f"    #{H[2]:20d}\n"
            s += f"  is_log  =  {bool(self.psp.is_log)}"
        else:
            s += "empty  =  True"
        return s

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


@warn_api()
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
