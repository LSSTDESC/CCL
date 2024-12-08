__all__ = ("Pk2D", "parse_pk2d", "parse_pk",)

import numpy as np

from . import (
    CCLObject, DEFAULT_POWER_SPECTRUM, UnlockInstance, check, get_pk_spline_a,
    get_pk_spline_lk, lib, unlock_instance)
from . import CCLWarning, CCLError, warnings
from .pyutils import _get_spline1d_arrays, _get_spline2d_arrays


class Pk2D(CCLObject):
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
        ``pk(k, a)``. This is vectorized in both ``k`` and ``a``.

    Args:
        a_arr (`array`): 
            An array holding values of the scale factor
        lk_arr (`array`):
            An array holding values of the natural logarithm of the wavenumber
            (in :math:`\\mathrm{Mpc}^{-1}`).
        pk_arr (`array`) :
            A 2-D array of shape ``(na, nk)``, of the values of the power spectrum
            at ``a_arr`` and ``lk_arr``. Input array could be flattened, provided
            that its size is ``nk*na``. The array can hold the values of the
            natural logarithm of the power spectrum, depending on the value of
            ``is_logp``. Users must ensure that the power spectrum is sufficiently
            well sampled (i.e. the resolution of `a_arr` and `lk_arr` is high enough
            to sample the main features in the power spectrum). CCL uses bicubic
            interpolation to evaluate the power spectrum at any intermediate point
            in k and a.
        extrap_order_lok (:obj:`int`):  ``{0, 1, 2}``.
            Extrapolation order to be used on k-values below the minimum
            the splines. Note that extrapolation is either in :math:`\\log(P(k))`
            or in :math:`P(k)`, depending on the value of ``is_logp``.
        extrap_order_hik (:obj:`int`) :  ``{0, 1, 2}``.
            Extrapolation order to be used on k-values above the maximum of the
            splines. Note that extrapolation is either in :math:`\\log(P(k))` or
            in :math:`P(k)`, depending on the value of ``is_logp``.
        is_logp (:obj:`bool`):
            If True, ``pkarr`` holds the natural logarithm of the
            power spectrum. Otherwise, the true value of the power spectrum is
            expected. If ``is_logp`` is ``True``, arrays are interpolated in
            log-space.

    .. automethod:: __call__
    """ # noqa E501
    from ._core.repr_ import build_string_Pk2D as __repr__

    def __init__(self, *, a_arr=None, lk_arr=None, pk_arr=None,
                 is_logp=True, extrap_order_lok=1, extrap_order_hik=2):
        # Make sure input makes sense
        if (a_arr is None) or (lk_arr is None) or (pk_arr is None):
            raise ValueError("If you do not provide a function, "
                             "you must provide arrays")

        # Check that `a` is a monotonically increasing array.
        if not (np.diff(a_arr) > 0).all():
            raise ValueError("Input scale factor array in `a_arr` is not "
                             "monotonically increasing.")

        pkflat = pk_arr.flatten()
        # Check dimensions make sense
        if len(pkflat) != len(a_arr)*len(lk_arr):
            raise ValueError("Size of input arrays is inconsistent")

        status = 0
        self.psp, status = lib.set_pk2d_new_from_arrays(lk_arr, a_arr, pkflat,
                                                        int(extrap_order_lok),
                                                        int(extrap_order_hik),
                                                        int(is_logp), status)
        check(status)

    @classmethod
    def from_function(cls, pkfunc, *, is_logp=True,
                      spline_params=None,
                      extrap_order_lok=1, extrap_order_hik=2):
        """ Generates a `Pk2D` object from a function that calculates a power
        spectrum.

        Args:
            pkfunc (:obj:`callable`):
                Function with signature ``f(k, a)`` which takes vectorized
                input in ``k`` (wavenumber in :math:`\\mathrm{Mpc}^{-1}`)
                and a scale factor`a``, and returns the value of the
                corresponding quantity. ``pkfunc`` will be sampled at the
                values of ``k`` and ``a`` used internally by CCL to store the
                linear and non-linear power spectra.
            spline_params (:obj:`~pyccl._core.parameters.parameters_base.SplineParameters`):
                optional spline parameters. Used to determine sampling rates
                in scale factor and wavenumber. Custom parameters can be passed
                via the :class:`~pyccl.cosmology.Cosmology` object with
                ``cosmo.cosmo.spline_params`` (C API), or with an instance of
                ``ccl.parameters.SplineParameters`` (Python API). If ``None``, it
                defaults to the global accuracy parameters in CCL at the moment
                this function is called.

        Returns:
            :class:`~pyccl.pk2d.Pk2D`. Power spectrum object.
        """ # noqa E501
        if spline_params is None:
            from . import spline_params
        # Set k and a sampling from CCL parameters
        a_arr = get_pk_spline_a(spline_params=spline_params)
        lk_arr = get_pk_spline_lk(spline_params=spline_params)

        # Compute power spectrum on 2D grid
        pk_arr = np.array([pkfunc(k=np.exp(lk_arr), a=a) for a in a_arr])

        return cls(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                   is_logp=is_logp, extrap_order_lok=extrap_order_lok,
                   extrap_order_hik=extrap_order_hik)

    def __eq__(self, other):
        # Check object id.
        if self is other:
            return True
        # Check the object class.
        if type(self) is not type(other):
            return False
        # If the objects contain no data, return early.
        if not (self or other):
            return True
        # Check extrapolation orders.
        if not (self.extrap_order_lok == other.extrap_order_lok
                and self.extrap_order_hik == other.extrap_order_hik):
            return False
        # Check the individual splines.
        a1, lk1, pk1 = self.get_spline_arrays()
        a2, lk2, pk2 = other.get_spline_arrays()
        return ((a1 == a2).all() and (lk1 == lk2).all()
                and np.array_equal(pk1, pk2))

    def __hash__(self):
        return hash(repr(self))

    @property
    def has_psp(self):
        return 'psp' in vars(self)

    @property
    def extrap_order_lok(self):
        return self.psp.extrap_order_lok if self else None

    @property
    def extrap_order_hik(self):
        return self.psp.extrap_order_hik if self else None

    @classmethod
    def from_model(cls, cosmo, model):
        """:class:`Pk2D` constructor returning the power spectrum
        associated with a given numerical model.

        Arguments:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            model (:obj:`str`): Model to use. These models allowed:

                  - ``'bbks'`` (`Bardeen et al. 1986 <https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B/abstract>`_).
                  - ``'eisenstein_hu'`` (`Eisenstein & Hu 1997 <https://arxiv.org/abs/astro-ph/9709112>`_).
                  - ``'eisenstein_hu_nowiggles'`` (`Eisenstein & Hu 1997 <https://arxiv.org/abs/astro-ph/9709112>`_, no-wiggles version).

        Returns:
            :class:`~pyccl.pk2d.Pk2D`. The power spectrum of the input model.
        """  # noqa E501

        pk2d = Pk2D.__new__(cls)
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
        else:
            raise ValueError(f"Invalid model {model}.")

        if np.ndim(ret) == 0:
            status = ret
        else:
            with UnlockInstance(pk2d):
                pk2d.psp, status = ret

        check(status, cosmo)
        return pk2d

    def apply_halofit(self, cosmo):
        """Apply the "HALOFIT" transformation of
        `Takahashi et al. 2012 <https://arxiv.org/abs/1208.2701>`_ on the linear
        power spectrum represented by this :class:`Pk2D` object, and return the
        result as a :class:`Pk2D` object.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.

        Return:
            :class:`Pk2D` object containing the non-linear power spectrum after
            applying HALOFIT.
        """ # noqa 501

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

        pk2d = Pk2D.__new__(Pk2D)
        status = 0
        ret = lib.apply_halofit(cosmo.cosmo, self.psp, status)
        if np.ndim(ret) == 0:
            status = ret
        else:
            with UnlockInstance(pk2d):
                pk2d.psp, status = ret
        check(status, cosmo)
        return pk2d

    def __call__(self, k, a, cosmo=None, *, derivative=False):
        """Evaluate the power spectrum or its logarithmic derivative at
        a single value of the scale factor.

        Arguments
        ---------
        k : :obj:`float` or `array`
            Wavenumber value(s) in units of :math:`{\\ rm Mpc}^{-1}`.
        a : :obj:`float` or `array`
            Value of the scale factor
        cosmo : :class:`~pyccl.cosmology.Cosmology`
            Cosmology object. Used to evaluate the power spectrum outside
            of the interpolation range in ``a``, thorugh the linear growth
            factor. If ``cosmo`` is ``None``, attempting to evaluate the power
            spectrum outside of the interpolation range will raise an error.
        derivative : :obj:`bool`
            If ``False``, evaluate the power spectrum. If ``True``, evaluate
            the logarithmic derivative of the power spectrum,
            :math:`d\\log P(k)/d\\log k`.

        Returns
        -------
        P(k, a) : (:obj:`float` or `array`)
            Value(s) of the power spectrum. or its derivative.
        """
        # determine if logarithmic derivative is needed
        if not derivative:
            eval_func = lib.pk2d_eval_multi
        else:
            eval_func = lib.pk2d_der_eval_multi

        # handle scale factor extrapolation
        if cosmo is None:
            cosmo = self.__call__._cosmo
            self.psp.extrap_linear_growth = 404  # flag no extrapolation
        else:
            cosmo.compute_growth()  # growth factors for extrapolation
            self.psp.extrap_linear_growth = 401  # flag extrapolation

        a_use = np.atleast_1d(a).astype(float)
        k_use = np.atleast_1d(k).astype(float)
        lk_use = np.log(k_use)

        status = 0
        out = np.zeros([len(a_use), len(k_use)])
        for ia, aa in enumerate(a_use):
            f, status = eval_func(self.psp, lk_use, aa,
                                  cosmo.cosmo, k_use.size, status)

            # Catch scale factor extrapolation bounds error.
            if status == lib.CCL_ERROR_SPLINE_EV:
                raise ValueError(
                    "Pk2D evaluation scale factor is outside of the "
                    "interpolation range. To extrapolate, pass a Cosmology.")
            check(status, cosmo)
            out[ia] = f

        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        return out

    # Save a dummy cosmology as an attribute of the `__call__` method
    # so we don't have to initialize one every time no `cosmo` is passed.
    # This is gentle with memory too, as `free` does not work for an empty
    # cosmology.
    __call__._cosmo = type("Dummy", (object,), {"cosmo": lib.cosmology()})()

    def copy(self):
        """Return a copy of this Pk2D object."""
        if not self:
            return Pk2D.__new__(Pk2D)
        return self + 0

    def get_spline_arrays(self):
        """Get the spline data arrays internally stored by this object to
        interpolate the power spectrum.

        Returns:
            Tuple containing

            - a_arr: Array of scale factors.
            - lk_arr: Array of natural logarithm of wavenumber k.
            - pk_arr: Array of the power spectrum :math:`P(k, a)`. The shape
              is ``(a_arr.size, lk_arr.size)``.
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
                f"{self.psp.amin} <= a <= {self.psp.amax}.",
                category=CCLWarning, importance='low')
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
                "power may lead to unexpected results",
                category=CCLWarning, importance='high')

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

    @unlock_instance
    def __iadd__(self, other):
        self = self + other
        return self

    @unlock_instance
    def __imul__(self, other):
        self = self * other
        return self

    @unlock_instance
    def __isub__(self, other):
        self = self - other
        return self

    @unlock_instance
    def __itruediv__(self, other):
        self = self / other
        return self

    @unlock_instance
    def __ipow__(self, other):
        self = self**other
        return self


def parse_pk2d(cosmo, p_of_k_a=DEFAULT_POWER_SPECTRUM, *, is_linear=False):
    """ Return the C-level `f2d` spline associated with a
    :class:`Pk2D` object.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        p_of_k_a (:class:`Pk2D` or :obj:`str`): if a
            :class:`Pk2D` object, its `f2d` spline will be used. If
            a string, the linear or non-linear power spectrum stored
            by ``cosmo`` under this name will be used. Defaults to the
            matter power spectrum stored in `cosmo`.
        is_linear (:obj:`bool`): if ``True``, and if ``p_of_k_a`` is a
            string or ``None``, the linear version of the corresponding
            power spectrum will be used (otherwise it'll be the
            non-linear version).
    """
    if isinstance(p_of_k_a, Pk2D):
        psp = p_of_k_a.psp
    else:
        if isinstance(p_of_k_a, str):
            name = p_of_k_a
        else:
            raise ValueError("p_of_k_a must be a pyccl.Pk2D object or "
                             "a string")

        if is_linear:
            cosmo.compute_linear_power()
            pk = cosmo.get_linear_power(name)
        else:
            cosmo.compute_nonlin_power()
            pk = cosmo.get_nonlin_power(name)
        psp = pk.psp
    return psp


def parse_pk(cosmo, p_of_k_a=None):
    """Helper to retrieve the right :class:`Pk2D` object.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`):
            3D Power spectrum to return. If a `Pk2D`, it is just returned.
            If `None` or `"linear"`, the linear power spectrum stored in
            ``cosmo`` is returned. If `"nonlinear"`, the nonlinear matter
            power spectrum stored in ``cosmo`` is returned.

    Returns:
        :class:`Pk2D` object corresponding to ``p_of_k_a``.
    """
    if not (p_of_k_a is None or isinstance(p_of_k_a, (str, Pk2D))):
        raise TypeError("p_of_k_a must be None, 'linear', 'nonlinear', Pk2D.")

    if isinstance(p_of_k_a, Pk2D):
        return p_of_k_a
    elif p_of_k_a is None or p_of_k_a == "linear":
        cosmo.compute_linear_power()
        return cosmo.get_linear_power()
    elif p_of_k_a == "nonlinear":
        cosmo.compute_nonlin_power()
        return cosmo.get_nonlin_power()
