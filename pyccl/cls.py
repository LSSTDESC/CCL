
import ccllib as lib
import constants as const
from pyutils import _cosmology_obj, check
import numpy as np

# Mapping between names for tracers and internal CCL tracer types
tracer_types = {
    'nc':               const.CL_TRACER_NC,
    'number_count':     const.CL_TRACER_NC,
    'wl':               const.CL_TRACER_WL,
    'lensing':          const.CL_TRACER_WL,
    'weak_lensing':     const.CL_TRACER_WL,
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
                 f_red=None):
        """
        Object handling a ClTracer (tracer with an angular power spectrum).

        Note: unless otherwise stated defaults are None.

        Args:
            cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object.
            tracer_type (:obj:`str`): Specifies what tracer to use.
            has_rsd (bool, optional): Flag to incorporate RSD into the model for CL. Defaults to False.
            has_magnification (bool, optional): Flag to incorporate magnification into the model for Cl. Defaults to False.
            has_intrinsic_alignment (bool, optional): Flag to incorporate intrinsic alignment into the model. Defaults to False.
            z (array_like, optional): Array of redshifts for N(z).
            n (array_like, optional): Array of N(z)-values.
            bias (array_like, optional): Array of alignment biases.
            mag_bias (array_like, optional): Array of shapes.
            bias_ia (array_like, optional): Array of intrinsic alignment amplitudes.
            f_red (array_like, optional): Array of red fractions.

        """
        # Verify cosmo object
        cosmo = _cosmology_obj(cosmo)
        
        # Check tracer type
        if tracer_type not in tracer_types.keys():
            raise KeyError("'%s' is not a valid tracer_type." % tracer_type)
        
        # Convert array arguments that are 'None' into 'NoneArr' type, and 
        # check whether arrays were specified as tuples or with a common z array
        z_n, n = _check_array_params(z, n, 'n')
        z_b, b = _check_array_params(z, bias, 'bias')
        z_s, s = _check_array_params(z, mag_bias, 'mag_bias')
        z_ba, ba = _check_array_params(z, bias_ia, 'bias_ia')
        z_rf, rf = _check_array_params(z, f_red, 'f_red')
        
        # Construct new ccl_cl_tracer
        status = 0
        self.cltracer, status = lib.cl_tracer_new_wrapper(
                            cosmo, 
                            tracer_types[tracer_type],
                            int(has_rsd), 
                            int(has_magnification), 
                            int(has_intrinsic_alignment),
                            z_n, n, z_b, b, z_s, s, z_ba, ba, z_rf, rf, 
                            status )
        
    def __del__(self):
        """Free memory associated with CCL_ClTracer object.

        """
        lib.cl_tracer_free(self.cltracer)


class ClTracerNumberCounts(ClTracer):
    """ClTracer for galaxy clustering.

    """
    
    def __init__(self, cosmo, has_rsd, has_magnification, 
                 n, bias, z=None, mag_bias=None):
        """ClTracer for galaxy clustering, N(z).

        Args:
            cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object.
            has_rsd (bool, optional): Flag to incorporate RSD into the model for CL. Defaults to False.
            has_magnification (bool, optional): Flag to incorporate magnification into the model for Cl. Defaults to False.
            z (array_like): Array of redshifts.
            n (array_like): Array of N(z)-values.
            bias (array_like): Array of biases.
            mag_bias (array_like, optional): Array of magnification bias. Not optional if has_magnification is True.

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
    """ClTracer for lensing shapes.

    """
    
    def __init__(self, cosmo, has_intrinsic_alignment, 
                 n, z=None, bias_ia=None, f_red=None):
        """ClTracer for lensing shapes.
        
        Args:
            cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object.
            has_intrinsic_alignment (bool, optional): Flag to incorporate intrinsic alignment into the model. Defaults to False.
            z (array_like): Array of redshifts.
            n (array_like): Array of N(z)-values.
            bias_ia (array_like, optional): Array of biases. Not optional if has_intrinsic_alignment is True.
            f_red (array_like, optional): Array of red fractions. Not optional if has_intrinsic_alignment is True.

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


def _cltracer_obj(cltracer):
    """Returns a CCL_ClTracer object, given an input object which may be 
    CCL_ClTracer, the ClTracer wrapper class, or an invalid type.

    Args:
        cltracer (:obj:): Either a CCL_ClTracer or the ClTracer wrapper class.

    Returns:
        cltracer (:obj:): Either a CCL_ClTracer or the ClTracer wrapper class.

    """
    if isinstance(cltracer, lib.CCL_ClTracer):
        return cltracer
    elif isinstance(cltracer, ClTracer):
        return cltracer.cltracer
    else:
        raise TypeError("Invalid ClTracer or CCL_ClTracer object.")


def _check_array_params(z, f_arg, f_name):
    """
    Check whether array arguments passed into the constructor of ClTracer() are 
    valid. If an array argument is set to 'None', it will be returned as 
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
    """Calculate angular power spectrum for two tracers.

    Args:
        cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object
        cltracer1 (:obj:): A Cl tracer.
        cltracer2 (:obj:): A Cl tracer.
        ell (float or array_like): Angular wavenumber to evaluate the tracers at.

    Returns:
        cl (float or array_like): Cl values for the tracers at `ell`.

    """
    # Access ccl_cosmology object
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
    check(status)
    return cl
