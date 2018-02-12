from pyccl import ccllib as lib
from pyccl import constants as const
from pyccl.pyutils import _cosmology_obj, check
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

# Define symbolic 'None' type for arrays, to allow proper handling by swig wrapper
NoneArr = np.array([])

class ClTracer(object):
    """ClTracer class used to wrap the cl_tracer found
    in CCL.

    A ClTracer is a data structure that contains all information
    describing the transfer functon of one tracer of the matter
    distribution.

    """

    def __init__(self, cosmo, tracer_type=None, has_rsd=False,
                 has_magnification=False, has_intrinsic_alignment=False,
                 z=None, n=None, bias=None, mag_bias=None, bias_ia=None,
                 f_red=None,z_source=1100.):
        """
        ClTracer is a class for handling tracers that have an angular power
        spectrum.

        Note: unless otherwise stated, defaults are None.

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            tracer_type (:obj:`str`): Specifies which type of tracer is being
                specified. Must be one of the types specified in the
                `tracer_types` dict in `cls.py`.
            has_rsd (bool, optional): Flag for whether the tracer has a
                redshift-space distortion term. Defaults to False.
            has_magnification (bool, optional): Flag for whether the tracer has
                a magnification term. Defaults to False.
            has_intrinsic_alignment (bool, optional): Flag for whether the
                tracer has an intrinsic alignment term. Defaults to False.
            z (array_like, optional): Array of redshifts that the following
                functions are sampled at. This is overriden if tuples of the
                form (z, fn(z)) are specified for those kwargs instead (this
                allows the functions to be sampled differently in z).
            n (array_like or tuple, optional): Array of N(z) sampled at the
                redshifts given in the z array, or a tuple of arrays (z, N(z)).
                The units are arbitrary; N(z) will be normalized to unity.
            bias (array_like or tuple, optional): Array of galaxy bias b(z)
                sampled at the redshifts given in the z array, or a tuple of
                arrays (z, b(z)).
            mag_bias (array_like or tuple, optional): Array of magnification
                bias s(z) sampled at the redshifts given in the z array, or a
                tuple of arrays (z, s(z)).
            bias_ia (array_like or tuple, optional): Array of intrinsic
                alignment amplitudes b_IA(z), or a tuple of arrays (z, b_IA(z)).
            f_red (array_like or tuple, optional): Array of red galaxy
                fractions f_red(z), or a tuple of arrays (z, f_red(z)).
            z_source (float, optional): Redshift of source plane for CMB lensing.
        """
        # Verify cosmo object
        cosmo = _cosmology_obj(cosmo)

        # Check tracer type
        if tracer_type not in tracer_types.keys():
            raise KeyError("'%s' is not a valid tracer_type." % tracer_type)

        # Convert array arguments that are 'None' into 'NoneArr' type, and
        # check whether arrays were specified as tuples or with a common z array
        self.z_n, self.n = _check_array_params(z, n, 'n')
        self.z_b, self.b = _check_array_params(z, bias, 'bias')
        self.z_s, self.s = _check_array_params(z, mag_bias, 'mag_bias')
        self.z_ba, self.ba = _check_array_params(z, bias_ia, 'bias_ia')
        self.z_rf, self.rf = _check_array_params(z, f_red, 'f_red')
        self.z_source = z_source

        # Construct new ccl_cl_tracer
        status = 0
        self.cltracer, status = lib.cl_tracer_new_wrapper(
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
                            self.z_source,
                            status )

    def __del__(self):
        """Free memory associated with CCL_ClTracer object.

        """
        lib.cl_tracer_free(self.cltracer)


class ClTracerNumberCounts(ClTracer):
    """
    ClTracer for galaxy number counts (galaxy clustering).
    """

    def __init__(self, cosmo, has_rsd, has_magnification,
                 n, bias, z=None, mag_bias=None):
        """
        ClTracer class for a tracer of galaxy number counts (galaxy clustering).

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            has_rsd (bool, optional): Flag for whether the tracer has a
                redshift-space distortion term. Defaults to False.
            has_magnification (bool, optional): Flag for whether the tracer has
                a magnification term. Defaults to False. mag_bias must be
                specified if set to True.
            z (array_like, optional): Array of redshifts that the following
                functions are sampled at. This is overriden if tuples of the
                form (z, fn(z)) are specified for those kwargs instead (this
                allows the functions to be sampled differently in z).
            n (array_like or tuple, optional): Array of N(z) sampled at the
                redshifts given in the z array, or a tuple of arrays (z, N(z)).
                The units are arbitrary; N(z) will be normalized to unity.
            bias (array_like or tuple, optional): Array of galaxy bias b(z)
                sampled at the redshifts given in the z array, or a tuple of
                arrays (z, b(z)).
            mag_bias (array_like or tuple, optional): Array of magnification
                bias s(z) sampled at the redshifts given in the z array, or a
                tuple of arrays (z, s(z)).
        """

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
    """
    ClTracer for weak lensing shear (galaxy shapes).
    """

    def __init__(self, cosmo, has_intrinsic_alignment,
                 n, z=None, bias_ia=None, f_red=None):
        """
        ClTracer class for a tracer of weak lensing shear (galaxy shapes).

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            has_intrinsic_alignment (bool, optional): Flag for whether the
                tracer has an intrinsic alignment term. Defaults to False.
                bias_ia and f_red must be specified if set to True.
            z (array_like, optional): Array of redshifts that the following
                functions are sampled at. This is overriden if tuples of the
                form (z, fn(z)) are specified for those kwargs instead (this
                allows the functions to be sampled differently in z).
            n (array_like or tuple, optional): Array of N(z) sampled at the
                redshifts given in the z array, or a tuple of arrays (z, N(z)).
                The units are arbitrary; N(z) will be normalized to unity.
            bias_ia (array_like or tuple, optional): Array of intrinsic
                alignment amplitudes b_IA(z), or a tuple of arrays (z, b_IA(z)).
            f_red (array_like or tuple, optional): Array of red galaxy
                fractions f_red(z), or a tuple of arrays (z, f_red(z)).
        """

        # Sanity check on input arguments
        if has_intrinsic_alignment \
        and (bias_ia is None or f_red is None):
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
    """
    ClTracer for CMB lensing.
    """

    def __init__(self, cosmo, z_source):
        """
        ClTracer class for a tracer of CMB lensing.

        Args:
            cosmo (:obj:`Cosmology`): Cosmology object.
            z_source (float): Redshift of source plane for CMB lensing.
        """

        # Call ClTracer constructor with appropriate arguments
        super(ClTracerCMBLensing, self).__init__(
                 cosmo=cosmo, tracer_type='cmbl',
                 has_rsd=False, has_magnification=False,
                 has_intrinsic_alignment=False,
                 z=None, n=None, bias=None, mag_bias=None,
                 bias_ia=None, f_red=None, z_source=z_source)


def _cltracer_obj(cltracer):
    """
    Returns a CCL_ClTracer object, given an input object which may be
    CCL_ClTracer, the ClTracer wrapper class, or an invalid type.

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
    """
    Check whether an argument `f_arg` passed into the constructor of ClTracer()
    is valid.

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
                                "Use %s=(z, %s), or pass the 'z' kwarg." \
                                % (f_name, f_name, f_name))
            z_f = np.atleast_1d(z)
            f = np.atleast_1d(f_arg)
    return z_f, f


def angular_cl(cosmo, cltracer1, cltracer2, ell):
    """
    Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        cltracer1, cltracer2 (:obj:): ClTracer objects, of any kind.
        ell (float or array_like): Angular wavenumber(s) to evaluate the
            angular power spectrum at.

    Returns:
        cl (float or array_like): Angular (cross-)power spectrum values, C_ell,
            for the pair of tracers, as a function of `ell`.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = _cosmology_obj(cosmo)

    # Access CCL_ClTracer objects
    clt1 = _cltracer_obj(cltracer1)
    clt2 = _cltracer_obj(cltracer2)

    status = 0
    # Return Cl values, according to whether ell is an array or not
    if isinstance(ell, float) or isinstance(ell, int) :
        # Use single-value function
        cl, status = lib.angular_cl(cosmo, ell, clt1, clt2, status)
    elif isinstance(ell, np.ndarray):
        # Use vectorised function
        cl, status = lib.angular_cl_vec(cosmo, clt1, clt2, ell, ell.size, status)
    else:
        # Use vectorised function
        cl, status = lib.angular_cl_vec(cosmo, clt1, clt2, ell, len(ell), status)
    check(status, cosmo_in)
    return cl
