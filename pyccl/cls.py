from . import ccllib as lib
from . import constants as const
from .core import _cosmology_obj, check
import numpy as np

# Mapping between names for tracers and internal CCL tracer types
tracer_types = {
    'nc':               const.CL_TRACER_NC,
    'number_count':     const.CL_TRACER_NC,
    'wl':               const.CL_TRACER_WL,
    'lensing':          const.CL_TRACER_WL,
    'weak_lensing':     const.CL_TRACER_WL,
    'cmbl':             const.CL_TRACER_CL,
    'cmb_lensing':      const.CL_TRACER_CL,
}

# Same mapping for non-Limber integration methods
nonlimber_methods = {
    'native': const.CCL_NONLIMBER_METHOD_NATIVE,
    'angpow': const.CCL_NONLIMBER_METHOD_ANGPOW,
}

function_types = {
    'dndz':             const.CCL_CLT_NZ,
    'bz':               const.CCL_CLT_BZ,
    'bias':             const.CCL_CLT_BZ,
    'sz':               const.CCL_CLT_SZ,
    'm_bias':           const.CCL_CLT_SZ,
    'rfz':              const.CCL_CLT_RF,
    'red_fraction':     const.CCL_CLT_RF,
    'baz':              const.CCL_CLT_BA,
    'a_bias':           const.CCL_CLT_BA,
    'wL':               const.CCL_CLT_WL,
    'window_lensing':   const.CCL_CLT_WL,
    'wM':               const.CCL_CLT_WM,
    'window_magnif':    const.CCL_CLT_WM,
}

# Define symbolic 'None' type for arrays, to allow proper handling by swig
# wrapper
NoneArr = np.array([])


class ClTracer(object):
    """A tracer of the matter density field.

    This class contains all information describing the transfer functon of
    a tracer (e.g., galaxy density, lensing shear) of the matter distribution.

    Args:
        cosmo (:obj:`Cosmology`): Cosmology object.
        tracer_type (:obj:`str`): Specifies which type of tracer is being
            specified. Must be one of
                'nc', 'number_count': number count tracer
                'wl', 'lensing', 'weak_lensing': lensing tracer
                'cmbl', 'cmb_lensing': CMB lensing tracer
        has_rsd (bool, optional): Flag for whether the tracer has a
            redshift-space distortion term. Defaults to False.
        has_magnification (bool, optional): Flag for whether the tracer has
            a magnification term. Defaults to False.
        has_intrinsic_alignment (bool, optional): Flag for whether the
            tracer has an intrinsic alignment term. Defaults to False.
        z (array_like, optional): Array of redshifts that the following
            functions are sampled at. This is overriden if tuples of the
            form (z, fn(z)) are specified for those kwargs instead (this
            allows the functions to be sampled differently in z). If `None`,
            then tuples for the other redshift dependent arguments are
            expected. Defaults to None.
        n (array_like or tuple, optional): Array of N(z) sampled at the
            redshifts given in the z array, or a tuple of arrays (z, N(z)).
            The units are arbitrary; N(z) will be normalized to unity. If
            `None`, the tracer is assumed to not have a redshift distribution
            (e.g., it has a single source source redshift like the CMB).
            Defaults to None.
        bias (array_like or tuple, optional): Array of galaxy bias b(z)
            sampled at the redshifts given in the z array, or a tuple of
            arrays (z, b(z)). If `None`, the tracer is assumbed to not
            have a bias parameter. Defaults to None.
        mag_bias (array_like or tuple, optional): Array of magnification
            bias s(z) sampled at the redshifts given in the z array, or a
            tuple of arrays (z, s(z)). If `None`, the tracer is assumed
            to not have magnification bias terms. Defaults to None.
        bias_ia (array_like or tuple, optional): Array of intrinsic
            alignment amplitudes b_IA(z), or a tuple of arrays
            (z, b_IA(z)). If `None`, the tracer is assumped to not have
            intrinsic alignments. Defaults to None.
        f_red (array_like or tuple, optional): Array of red galaxy
            fractions f_red(z), or a tuple of arrays (z, f_red(z)).
            If `None`, then the tracer is assumed to not have a red fraction.
            Defaults to None.
        z_source (float, optional): Redshift of source plane for CMB
            lensing. Defaults to 1100.
    """

    def __init__(self, cosmo, tracer_type, has_rsd=False,
                 has_magnification=False, has_intrinsic_alignment=False,
                 z=None, n=None, bias=None, mag_bias=None, bias_ia=None,
                 f_red=None, z_source=1100.):
        # Verify cosmo object
        cosmo = _cosmology_obj(cosmo)

        # Check tracer type
        if tracer_type not in tracer_types.keys():
            raise ValueError("'%s' is not a valid tracer_type." % tracer_type)

        # Convert array arguments that are 'None' into 'NoneArr' type, and
        # check whether arrays were specified as tuples or with a common z
        # array
        self.z_n, self.n = _check_array_params(z, n, 'n')
        self.z_b, self.b = _check_array_params(z, bias, 'bias')
        self.z_s, self.s = _check_array_params(z, mag_bias, 'mag_bias')
        self.z_ba, self.ba = _check_array_params(z, bias_ia, 'bias_ia')
        self.z_rf, self.rf = _check_array_params(z, f_red, 'f_red')
        self.z_source = z_source

        # Construct new ccl_cl_tracer
        status = 0
        return_val = lib.cl_tracer_new_wrapper(
                            cosmo,
                            tracer_types[tracer_type],
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
                    'bz', 'bias': bias
                    'sz', 'm_bias': magnification bias
                    'rfz', 'red_fraction': red fraction
                    'baz', 'a_bias': intrinsic alignment bias
                    'wL', 'window_lensing': weak lensing window function
                    'wM', 'window_magnif': magnification window function
            a (:obj: float or array-like): list of scale factors at which to
                evaluate the function.

        Returns:
            Array of function values at the input scale factors.
        """
        # Access ccl_cosmology object
        cosmo_in = cosmo
        cosmo = _cosmology_obj(cosmo)

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


class ClTracerNumberCounts(ClTracer):
    """ClTracer for galaxy number counts (galaxy clustering).

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            has_rsd (bool): Flag for whether the tracer has a
                redshift-space distortion term.
            has_magnification (bool): Flag for whether the tracer has
                a magnification term. mag_bias must be
                specified if set to True.
            n (array_like or tuple): Array of N(z) sampled at the
                redshifts given in the z array, or a tuple of arrays (z, N(z)).
                The units are arbitrary; N(z) will be normalized to unity.
            bias (array_like or tuple): Array of galaxy bias b(z)
                sampled at the redshifts given in the z array, or a tuple of
                arrays (z, b(z)).
            z (array_like, optional): Array of redshifts that the following
                functions are sampled at. This is overriden if tuples of the
                form (z, fn(z)) are specified for those kwargs instead (this
                allows the functions to be sampled differently in z).
                Defaults to None.
            mag_bias (array_like or tuple, optional): Array of magnification
                bias s(z) sampled at the redshifts given in the z array, or a
                tuple of arrays (z, s(z)). Defaults to None for no
                magnification bias.
    """

    def __init__(self, cosmo, has_rsd, has_magnification,
                 n, bias, z=None, mag_bias=None):
        # Sanity check on input arguments
        if has_magnification and mag_bias is None:
                raise ValueError("Keyword arg 'mag_bias' must be specified if "
                                 "has_magnification=True.")

        # Call ClTracer constructor with appropriate arguments
        super(ClTracerNumberCounts, self).__init__(
                 cosmo=cosmo, tracer_type='nc',
                 has_rsd=has_rsd, has_magnification=has_magnification,
                 has_intrinsic_alignment=False,
                 z=z, n=n, bias=bias, mag_bias=mag_bias,
                 bias_ia=None, f_red=None)


class ClTracerLensing(ClTracer):
    """ClTracer for weak lensing shear (galaxy shapes).

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            has_intrinsic_alignment (bool): Flag for whether the
                tracer has an intrinsic alignment term.
                bias_ia and f_red must be specified if set to True.
            n (array_like or tuple): Array of N(z) sampled at the
                redshifts given in the z array, or a tuple of arrays (z, N(z)).
                The units are arbitrary; N(z) will be normalized to unity.
            z (array_like, optional): Array of redshifts that the following
                functions are sampled at. This is overriden if tuples of the
                form (z, fn(z)) are specified for those kwargs instead (this
                allows the functions to be sampled differently in z).
                Defaults to None.
            bias_ia (array_like or tuple, optional): Array of intrinsic
                alignment amplitudes b_IA(z), or a tuple of arrays
                (z, b_IA(z)). Defaults to None.
            f_red (array_like or tuple, optional): Array of red galaxy
                fractions f_red(z), or a tuple of arrays (z, f_red(z)).
                Defaults to None.
    """

    def __init__(self, cosmo, has_intrinsic_alignment,
                 n, z=None, bias_ia=None, f_red=None):
        # Sanity check on input arguments
        if (has_intrinsic_alignment and
                (bias_ia is None or f_red is None)):
            raise ValueError("Keyword args 'bias_ia' and 'f_red' must be "
                             "specified if has_intrinsic_alignment=True.")

        # Call ClTracer constructor with appropriate arguments
        super(ClTracerLensing, self).__init__(
                 cosmo=cosmo, tracer_type='wl',
                 has_rsd=False, has_magnification=False,
                 has_intrinsic_alignment=has_intrinsic_alignment,
                 z=z, n=n, bias=None, mag_bias=None,
                 bias_ia=bias_ia, f_red=f_red)


class ClTracerCMBLensing(ClTracer):
    """ClTracer for CMB lensing.

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            z_source (float): Redshift of source plane for CMB lensing.
    """

    def __init__(self, cosmo, z_source):
        # Call ClTracer constructor with appropriate arguments
        super(ClTracerCMBLensing, self).__init__(
                 cosmo=cosmo, tracer_type='cmbl',
                 has_rsd=False, has_magnification=False,
                 has_intrinsic_alignment=False,
                 z=None, n=None, bias=None, mag_bias=None,
                 bias_ia=None, f_red=None, z_source=z_source)


def _cltracer_obj(cltracer):
    """Returns a CCL_ClTracer object from an CCL_ClTracer or
    the ClTracer wrapper classself.

    Invalid input raises a TypeError.

    Args:
        cltracer (:obj:): Either a CCL_ClTracer or the ClTracer wrapper class.

    Returns:
        cltracer (:obj:): A CCL_ClTracer that can be passed out to the CCL C
        library.
    """
    if isinstance(cltracer, lib.CCL_ClTracer):
        return cltracer
    elif isinstance(cltracer, ClTracer):
        return cltracer.cltracer
    else:
        raise TypeError("Invalid ClTracer or CCL_ClTracer object.")


def _check_array_params(z, f_arg, f_name):
    """Check whether an argument `f_arg` passed into the constructor of
    ClTracer() is valid.

    If the argument is set to `None`, it will be replaced with a special array
    that signals to the CCL wrapper that this argument is NULL.

    If the argument is given as an array, the redshift array passed to the CCL
    wrapper will be the `z` argument passed to this function.

    If the argument is given as a tuple of the form (z, fn(z)), the redshift
    array passed to the CCL wrapper will be the one from the tuple, and *not*
    the `z` argument passed to this function.
    """
    if f_arg is None:
        # Return empty array if argument is None
        f = NoneArr
        z_f = NoneArr
    else:
        if len(f_arg) == 2:
            # Redshift and function arrays were both specified
            z_f, f = f_arg
        else:
            # Only a function array was specified; redshifts must be given in
            # the 'z' array or an error is thrown.
            if z is None:
                raise TypeError("'%s' was specified without a redshift array. "
                                "Use %s=(z, %s), or pass the 'z' kwarg."
                                % (f_name, f_name, f_name))
            z_f = np.atleast_1d(np.array(z, dtype=float))
            f = np.atleast_1d(np.array(f_arg, dtype=float))
    return z_f, f


def angular_cl(cosmo, cltracer1, cltracer2, ell,
               l_limber=-1., l_logstep=1.05, l_linstep=20., dchi=3.,
               dlk=0.003, zmin=0.05, non_limber_method="native"):
    """
    Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        cltracer1, cltracer2 (:obj:): ClTracer objects, of any kind.
        ell (float or array_like): Angular wavenumber(s) at which to evaluate
            the angular power spectrum.
        l_limber (float) : Angular wavenumber beyond which Limber's
            approximation will be used. Defaults to 1.
        l_logstep (float) : logarithmic step in ell at low multipoles.
            Defaults to 1.05.
        l_linstep (float) : linear step in ell at high multipoles.
            Defaults to 20.
        dchi (float) : comoving distance step size in non-limber native
            integrals. Defaults to 3.
        dlk (float) : logarithmic step for the k non-limber native integral.
            Defaults to 0.003.
        zmin (float) : minimal redshift for the integrals. Defualts to 0.05.
        non_limber_method (str) : non-Limber integration method. Supported:
            "native" and "angpow". Defaults to 'native'.

    Returns:
        float or array_like: Angular (cross-)power spectrum values,
            :math:`C_\ell`, for the pair of tracers, as a function of
            :math:`\ell`.
    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)

    if non_limber_method not in nonlimber_methods.keys():
        raise ValueError(
            "'%s' is not a valid non-Limber integration method." %
            non_limber_method)

    # Access CCL_ClTracer objects
    clt1 = _cltracer_obj(cltracer1)
    clt2 = _cltracer_obj(cltracer2)

    status = 0
    # Return Cl values, according to whether ell is an array or not
    if isinstance(ell, float) or isinstance(ell, int):
        # Use single-value function
        cl_one, status = lib.angular_cl_vec(
            cosmo, clt1, clt2, l_limber, l_logstep, l_linstep, dchi, dlk, zmin,
            nonlimber_methods[non_limber_method], [ell], 1, status)
        cl = cl_one[0]
    elif isinstance(ell, np.ndarray):
        # Use vectorised function
        cl, status = lib.angular_cl_vec(
            cosmo, clt1, clt2, l_limber, l_logstep, l_linstep, dchi, dlk, zmin,
            nonlimber_methods[non_limber_method], ell, ell.size, status)
    else:
        # Use vectorised function
        cl, status = lib.angular_cl_vec(
            cosmo, clt1, clt2, l_limber, l_logstep, l_linstep, dchi, dlk, zmin,
            nonlimber_methods[non_limber_method], ell, len(ell), status)
    check(status)
    return cl
