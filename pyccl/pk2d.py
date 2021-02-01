from . import ccllib as lib

from .pyutils import check
import numpy as np


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
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object. The cosmology
             object is needed in order if `pkfunc` is not `None`. The object is
             used to determine the sampling rate in scale factor and
             wavenumber.
        empty (bool): if True, just create an empty object, to be filled
            out later
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
    def pk_from_model(Pk2D, cosmo, model):
        """`Pk2D` constructor returning the power spectrum associated with
        a given numerical model.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            model (:obj:`str`): model to use. Three models allowed:
                `'bbks'` (Bardeen et al. ApJ 304 (1986) 15).
                `'eisenstein_hu'` (Eisenstein & Hu astro-ph/9710252).
                `'emu'` (arXiv:1508.02654).
        """
        pk2d = Pk2D(empty=True)
        status = 0
        if model == 'bbks':
            cosmo.compute_growth()
            ret = lib.compute_linpower_bbks(cosmo.cosmo, status)
        elif model == 'eisenstein_hu':
            cosmo.compute_growth()
            ret = lib.compute_linpower_eh(cosmo.cosmo, status)
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

    def __del__(self):
        """Free memory associated with this Pk2D structure
        """
        if hasattr(self, 'has_psp'):
            if self.has_psp and hasattr(self, 'psp'):
                lib.f2d_t_free(self.psp)


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
