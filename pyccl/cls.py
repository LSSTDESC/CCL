from . import ccllib as lib
from .core import check
import numpy as np
import collections

function_types = {
    'dndz': lib.trf_nz,
    'bias': lib.trf_bz,
    'mag_bias': lib.trf_sz,
    'red_frac': lib.trf_rf,
    'ia_bias': lib.trf_ba,
    'lensing_win': lib.trf_wL,
    'mag_win': lib.trf_wM,
}

# Define symbolic 'None' type for arrays, to allow proper handling by swig
# wrapper
NoneArr = np.array([])


class Tracer(object):
    """A tracer of the matter density field.

    .. note:: This class cannot be used directly. Use one of
              :obj:`NumberCountsTracer`, :obj:`WeakLensingTracer`
              or :obj:`CMBLensingTracer` instead.

    This class contains all information describing the transfer functon of
    a tracer (e.g., galaxy density, lensing shear) of the matter distribution.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "A `Tracer` object cannot be used directly. Use one of "
            "`NumberCountsTracer`, `WeakLensingTracer` or `CMBLensingTracer` "
            "instead.")

    def _build_tracer(
            self, cosmo, tracer_type, has_rsd=False,
            dndz=None, bias=None, mag_bias=None, ia_bias=None,
            red_frac=None, z_source=1100.):
        """Build the CCL_ClTracer.

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            tracer_type (:obj:): Specifies the type of tracer. Must be one of
                    lib.number_counts_tracer: number count tracer
                    lib.weak_lensing_tracer: lensing tracer
                    lib.cmb_lensing_tracer: CMB lensing tracer
            has_rsd (bool, optional): Flag for whether the tracer has a
                redshift-space distortion term. Defaults to False.
            dndz (tuple of arrays, optional): A tuple of arrays (z, N(z))
                giving the redshift distribution of the objects. The units are
                arbitrary; N(z) will be normalized to unity. If `None`, the
                tracer is assumed to not have a redshift distribution (e.g.,
                it has a single source source redshift like the CMB). Defaults
                to None.
            bias (tuple of arrays, optional): A tuple of arrays (z, b(z))
                giving the galaxy bias. If `None`, the tracer is assumbed to
                not have a bias parameter. Defaults to None.
            mag_bias (tuple of arrays, optional): A tuple of arrays (z, s(z))
                giving the magnification bias as a function of redshift. If
                `None`, the tracer is assumed to not have magnification bias
                terms. Defaults to None.
            ia_bias (tuple of arrays, optional): A tuple of arrays
                (z, b_IA(z)) giving the intrinsic alignment amplitude b_IA(z).
                If `None`, the tracer is assumped to not have intrinsic
                alignments. Defaults to None.
            red_frac (tuple of arrays,, optional): A tuple of arrays
                (z, f_red(z)) givng the red fraction of galaxies as a function
                of redshift. If `None`, then the tracer is assumed to not have
                a red fraction. Defaults to None.
            z_source (float, optional): Redshift of source plane for CMB
                lensing. Defaults to 1100.
        """

        # Verify cosmo object
        cosmo = cosmo.cosmo

        has_magnification = mag_bias is not None
        if (red_frac is None) != (ia_bias is None):
            raise ValueError(
                "Either both or none of `red_frac` and `ia_bias` "
                "must be specified.")
        has_intrinsic_alignment = red_frac is not None

        # Passing None for certain arguments causes segmentation faults at the
        # moment. The following checks try to guard against these instances
        # but this should probably be checked for at the C level.
        if tracer_type in [lib.weak_lensing_tracer,
                           lib.number_counts_tracer]:
            if not isinstance(dndz, collections.Iterable) \
               or len(dndz) != 2 \
               or not (isinstance(dndz[0], collections.Iterable)
                       and isinstance(dndz[1], collections.Iterable)):
                raise ValueError("dndz needs to be a tuple of two arrays.")
        if tracer_type in [lib.number_counts_tracer]:
            if not isinstance(bias, collections.Iterable) \
               or len(bias) != 2 \
               or not (isinstance(bias[0], collections.Iterable)
                       and isinstance(bias[1], collections.Iterable)):
                raise ValueError("bias needs to be a tuple of two arrays.")

        # Convert array arguments that are 'None' into 'NoneArr' type and
        # check whether arrays were specified as tuples
        self.z_n, self.n = _check_array_params(dndz)
        self.z_b, self.b = _check_array_params(bias)
        self.z_s, self.s = _check_array_params(mag_bias)
        self.z_ba, self.ba = _check_array_params(ia_bias)
        self.z_rf, self.rf = _check_array_params(red_frac)
        self.z_source = z_source

        # Construct new ccl_cl_tracer
        status = 0
        return_val = lib.cl_tracer_new_wrapper(
                            cosmo,
                            tracer_type,
                            int(has_rsd),
                            int(has_magnification),
                            int(has_intrinsic_alignment),
                            self.z_n, self.n,
                            self.z_b, self.b,
                            self.z_s, self.s,
                            self.z_ba, self.ba,
                            self.z_rf, self.rf,
                            float(self.z_source),
                            status)

        if (isinstance(return_val, int)):
            self.has_cltracer = False
            check(return_val)
        else:
            self.has_cltracer = True
            self.cltracer, status = return_val

    def get_internal_function(self, cosmo, function, a):
        """
        Method to evaluate any internal function of redshift for this tracer.

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            function (:obj:`str`): Specifies which function to evaluate. Must
                be one of
                    'dndz': number density
                    'bias': bias
                    'mag_bias': magnification bias
                    'red_frac': red fraction
                    'ia_bias': intrinsic alignment bias
                    'lensing_win': weak lensing window function
                    'mag_win': magnification window function
            a (:obj: float or array-like): list of scale factors at which to
                evaluate the function.

        Returns:
            Array of function values at the input scale factors.
        """
        # Access ccl_cosmology object
        cosmo_in = cosmo
        cosmo = cosmo.cosmo

        # Check that specified function type exists
        if function not in function_types.keys():
            raise ValueError(
                "Internal function type '%s' not recognized."
                % function)

        # Check input types
        status = 0
        is_scalar = False
        if isinstance(a, float):
            is_scalar = True
            aarr = np.array([a])
            na = 1
        elif isinstance(a, np.ndarray):
            aarr = a
            na = a.size
        else:
            aarr = a
            na = len(a)

        # Evaluate function
        farr, status = lib.clt_fa_vec(cosmo, self.cltracer,
                                      function_types[function],
                                      aarr, na, status)
        check(status, cosmo_in)
        if is_scalar:
            return farr[0]
        else:
            return farr

    def __del__(self):
        """Free memory associated with CCL_ClTracer object.
        """
        if hasattr(self, 'has_cltracer'):
            if self.has_cltracer:
                lib.cl_tracer_free(self.cltracer)


class NumberCountsTracer(Tracer):
    """A Tracer for galaxy number counts (galaxy clustering).

    Args:
        cosmo (:obj:`Cosmology`): Cosmology object.
        has_rsd (bool): Flag for whether the tracer has a
            redshift-space distortion term.
        dndz (tuple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects. The units are
            arbitrary; N(z) will be normalized to unity.
        bias (tuple of arrays): A tuple of arrays (z, b(z))
            giving the galaxy bias.
        mag_bias (tuple of arrays, optional): A tuple of arrays (z, s(z))
            giving the magnification bias as a function of redshift. If
            `None`, the tracer is assumed to not have magnification bias
            terms. Defaults to None.
    """

    def __init__(self, cosmo, has_rsd, dndz, bias, mag_bias=None):
        # Call Tracer constructor with appropriate arguments
        self._build_tracer(
            cosmo=cosmo, tracer_type=lib.number_counts_tracer,
            has_rsd=has_rsd,
            dndz=dndz, bias=bias, mag_bias=mag_bias,
            ia_bias=None, red_frac=None)


class WeakLensingTracer(Tracer):
    """A Tracer for weak lensing shear (galaxy shapes).

    Args:
        cosmo (:obj:`Cosmology`): Cosmology object.
        dndz (tuple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects. The units are
            arbitrary; N(z) will be normalized to unity.
        ia_bias (tuple of arrays, optional): A tuple of arrays
            (z, b_IA(z)) giving the intrinsic alignment amplitude b_IA(z).
            If `None`, the tracer is assumped to not have intrinsic
            alignments. Defaults to None.
        red_frac (tuple of arrays,, optional): A tuple of arrays
            (z, f_red(z)) givng the red fraction of galaxies as a function
            of redshift. If `None`, then the tracer is assumed to not have
            a red fraction. Defaults to None.
    """

    def __init__(self, cosmo, dndz, ia_bias=None, red_frac=None):
        # Call Tracer constructor with appropriate arguments
        self._build_tracer(
            cosmo=cosmo, tracer_type=lib.weak_lensing_tracer,
            has_rsd=False,
            dndz=dndz, bias=None, mag_bias=None,
            ia_bias=ia_bias, red_frac=red_frac)


class CMBLensingTracer(Tracer):
    """A Tracer for CMB lensing.

    Args:
        cosmo (:obj:`Cosmology`): Cosmology object.
        z_source (float): Redshift of source plane for CMB lensing.
    """

    def __init__(self, cosmo, z_source):
        # Call Tracer constructor with appropriate arguments
        self._build_tracer(
            cosmo=cosmo, tracer_type=lib.cmb_lensing_tracer,
            has_rsd=False,
            dndz=None, bias=None, mag_bias=None,
            ia_bias=None, red_frac=None, z_source=z_source)


def _check_array_params(f_arg):
    """Check whether an argument `f_arg` passed into the constructor of
    Tracer() is valid.

    If the argument is set to `None`, it will be replaced with a special array
    that signals to the CCL wrapper that this argument is NULL.
    """
    if f_arg is None:
        # Return empty array if argument is None
        f = NoneArr
        z_f = NoneArr
    else:
        z_f = np.atleast_1d(np.array(f_arg[0], dtype=float))
        f = np.atleast_1d(np.array(f_arg[1], dtype=float))
    return z_f, f


def angular_cl(cosmo, cltracer1, cltracer2, ell,
               l_limber=-1., l_logstep=1.05, l_linstep=20.):
    """Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        cltracer1, cltracer2 (:obj:`Tracer`): Tracer objects, of any kind.
        ell (float or array_like): Angular wavenumber(s) at which to evaluate
            the angular power spectrum.
        l_limber (float) : Angular wavenumber beyond which Limber's
            approximation will be used. Defaults to -1.
        l_logstep (float) : logarithmic step in ell at low multipoles.
            Defaults to 1.05.
        l_linstep (float) : linear step in ell at high multipoles.
            Defaults to 20.

    Returns:
        float or array_like: Angular (cross-)power spectrum values,
            :math:`C_\\ell`, for the pair of tracers, as a function of
            :math:`\\ell`.
    """
    # Access ccl_cosmology object
    cosmo = cosmo.cosmo

    # Access CCL_ClTracer objects
    clt1 = cltracer1.cltracer
    clt2 = cltracer2.cltracer

    status = 0
    # Return Cl values, according to whether ell is an array or not
    if isinstance(ell, float) or isinstance(ell, int):
        # Use single-value function
        cl_one, status = lib.angular_cl_vec(
            cosmo, clt1, clt2, l_limber, l_logstep,
            l_linstep, [ell], 1, status)
        cl = cl_one[0]
    elif isinstance(ell, np.ndarray):
        # Use vectorised function
        cl, status = lib.angular_cl_vec(
            cosmo, clt1, clt2, l_limber, l_logstep,
            l_linstep, ell, ell.size, status)
    else:
        # Use vectorised function
        cl, status = lib.angular_cl_vec(
            cosmo, clt1, clt2, l_limber, l_logstep,
            l_linstep, ell, len(ell), status)
    check(status)
    return cl
