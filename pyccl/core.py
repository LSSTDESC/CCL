
import ccllib as lib
import numpy as np
from warnings import warn
from pyutils import check

# Configuration types
transfer_function_types = {
    'none':             lib.none,
    'emulator':         lib.emulator,
    'fitting_function': lib.fitting_function,
    'eisenstein_hu':    lib.eisenstein_hu,
    'bbks':             lib.bbks,
    'boltzmann':        lib.boltzmann,
    'boltzmann_camb':   lib.boltzmann_camb,
    'camb':             lib.boltzmann_camb,
    'boltzmann_class':  lib.boltzmann_class,
    'class':            lib.boltzmann_class
}

matter_power_spectrum_types = {
    'halo_model':   lib.halo_model,
    'halomodel':    lib.halo_model,
    'halofit':      lib.halofit,
    'linear':       lib.linear
}

mass_function_types = {
    'angulo':   lib.angulo,
    'tinker':   lib.tinker,
    'tinker10': lib.tinker10,
    'watson':   lib.watson
}

# Error types
error_types = {
    lib.CCL_ERROR_MEMORY:       'CCL_ERROR_MEMORY',
    lib.CCL_ERROR_LINSPACE:     'CCL_ERROR_LINSPACE',
    lib.CCL_ERROR_INCONSISTENT: 'CCL_ERROR_INCONSISTENT',
    lib.CCL_ERROR_SPLINE:       'CCL_ERROR_SPLINE',
    lib.CCL_ERROR_SPLINE_EV:    'CCL_ERROR_SPLINE_EV',
    lib.CCL_ERROR_INTEG:        'CCL_ERROR_INTEG',
    lib.CCL_ERROR_ROOT:         'CCL_ERROR_ROOT',
    lib.CCL_ERROR_CLASS:        'CCL_ERROR_CLASS'
}


class Parameters(object):
    """The Parameters class contains cosmological parameters.

    """
    
    def __init__(self, Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
                 Omega_k=0., Omega_n=0., w0=-1., wa=0., sigma8=None,
                 zarr_mgrowth=None, dfarr_mgrowth=None):
        """Creates a set of cosmological parameters.

        Note:
            Although some arguments default to `None`, they will raise
            a ValueError inside this function, so they are not optional.
        
        Args:
            Omega_c (float): Cold dark matter density fraction.
            Omega_b (float): Baryonic matter density fraction.
            h (float): Hubble constant divided by 100 km/s/Mpc; unitless.
            A_s (float): Power spectrum normalization; Mpc^-3 CHECKTHIS - PHIL BULL. Optional if sigma8 is specified.
            n_s (float): Power spectrum index.
            Omega_k (float, optional): Curvature density fraction. Defaults to 0.
            Omega_n (float, optional): Massless neutrino density fracton. Defaults to 0.
            w0 (float, optional): First order term of dark energy equation of state. Defaults to -1.
            wa (float, optional): Second order term of dark energy equation of state. Defaults to 0.
            sigma8 (float): Mass variance at 8 Mpc scale. Optional if A_s is specified.
            zarr_mgrowth (:obj: list of floats): UNKNOWN - PHIL BULL.
            dfarr_mgrowth (UNKNOWN): UNKNOWN - PHIL BULL.

        """
        # Set current ccl_parameters object to None
        self.parameters = None
        
         # Set nz_mgrowth (no. of redshift bins for modified growth fns.)
        if zarr_mgrowth is not None and dfarr_mgrowth is not None:
            # Get growth array size and do sanity check
            zarr_mgrowth = np.atleast_1d(zarr_mgrowth)
            dfarr_mgrowth = np.atleast_1d(dfarr_mgrowth)
            assert zarr_mgrowth.size == dfarr_mgrowth.size
            nz_mgrowth = zarr_mgrowth.size
        else:
            # If one or both of the MG growth arrays are set to zero, disable 
            # all of them
            if zarr_mgrowth is not None:
                warn("zarr_mgrowth ignored; must also specify dfarr_mgrowth.",
                     UserWarning)
            if dfarr_mgrowth is not None:
                warn("dfarr_mgrowth ignored; must also specify zarr_mgrowth.",
                     UserWarning)
            zarr_mgrowth = None
            dfarr_mgrowth = None
            nz_mgrowth = -1
        
        # Check to make sure specified amplitude parameter is consistent
        if (A_s is None and sigma8 is None) \
        or (A_s is not None and sigma8 is not None):
            raise ValueError("Must set either A_s or sigma8.")
        
        # Set norm_pk to either A_s or sigma8
        norm_pk = A_s if A_s is not None else sigma8
        
        # The C library decides whether A_s or sigma8 was the input parameter 
        # based on value, so we need to make sure this is consistent too
        if norm_pk >= 1e-5 and A_s is not None:
            raise ValueError("A_s must be less than 1e-5.")
            
        if norm_pk < 1e-5 and sigma8 is not None:
            raise ValueError("sigma8 must be greater than 1e-5.")
        
        # Check if any compulsory parameters are not set
        compul = [Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, norm_pk, n_s]
        names = ['Omega_c', 'Omega_b', 'Omega_k', 'Omega_n', 'w0', 'wa', 
                 'h', 'norm_pk', 'n_s']
        for nm, item in zip(names, compul):
            if item is None:
                raise ValueError("Necessary parameter '%s' was not set "
                                 "(or set to None)." % nm)
        
        # Create new instance of ccl_parameters object
        if nz_mgrowth == -1:
            # Create ccl_parameters without modified growth
            self.parameters = lib.parameters_create(
                                    Omega_c, Omega_b, Omega_k, Omega_n, 
                                    w0, wa, h, norm_pk, n_s, 
                                    -1, None, None)
        else:
            # Create ccl_parameters with modified growth arrays
            self.parameters = lib.parameters_create_vec(
                                    Omega_c, Omega_b, Omega_k, Omega_n, 
                                    w0, wa, h, norm_pk, n_s, 
                                    zarr_mgrowth, dfarr_mgrowth)
    
    def __getitem__(self, key):
        """Access parameter values by name.

        """
        try:
            val = getattr(self.parameters, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        return val
    
    def __setitem__(self, key, val):
        """Set parameter values by name.

        """
        raise NotImplementedError("Parameters objects are immutable; create a "
                                  "new Parameters() instance instead.")
        
        try:
            # First check if the key already exists (otherwise the parameter 
            # would be silently added to the ccl_parameters class instance)
            getattr(self.parameters, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        
        # Set value of parameter
        setattr(self.parameters, key, val)
        # TODO: Should update/replace CCL objects appropriately
    
    def __str__(self):
        """Output the parameters that were set, and their values.

        """
        params = ['Omega_c', 'Omega_b', 'Omega_m', 'Omega_n', 'Omega_k', 
                  'w0', 'wa', 'H0', 'h', 'A_s', 'n_s', 'Omega_g', 'T_CMB', 
                  'sigma_8', 'Omega_l', 'z_star', 'has_mgrowth']
  
        vals = ["%15s: %s" % (p, getattr(self.parameters, p)) for p in params]
        string = "Parameters\n----------\n"
        string += "\n".join(vals)
        return string


class Cosmology(object):
    """Wrapper for the ccl_cosmology object.

    Includes cosmological parameters and cached data.

    """
    
    def __init__(self, params, config=None, 
                 transfer_function='boltzmann_class',
                 matter_power_spectrum='halofit',
                 mass_function='tinker'):
        """Creates a wrapper for ccl_cosmology.

        TODO: enumerate transfer_function and 
        matter_power_spectrum options.

        Args:
            params (:obj:`Parameters`): Cosmological parameters object.
            config (:obj:`ccl_configuration`, optional): Configuration for how to use CCL. Takes precident over any other passed in configuration. Defaults to None.
            transfer_function (:obj:`str`, optional): The transfer function to use. Defaults to `boltzmann_class`.
            matter_power_spectrum (:obj:`str`, optional): The matter power spectrum to use. Defaults to `halofit`.
            mass_function (:obj:`str`, optional): The mass function to use. Defaults to `tinker` (2010).

        """
        # Check the type of the input params object
        if isinstance(params, lib.parameters):
            self.params = {} # Set to empty dict if ccl_parameters given directly
        elif isinstance(params, Parameters):
            self.params = params
            params = params.parameters # We only need the ccl_parameters object
        else:
            raise TypeError("'params' is not a valid ccl_parameters or "
                            "Parameters object.")
        
        # Check that the ccl_configuration-related arguments are valid
        if config is not None:
            # User passed a ccl_configuration object; ignore other arguments 
            # and use this
            
            # Check that input object is of the correct type
            if not isinstance(config, lib.configuration):
                raise TypeError("'config' is not a valid ccl_configuration "
                                "object.")
            
            # Store ccl_configuration for later access
            self.configuration = config
            
        else:
            # Construct a new ccl_configuration object from kwargs
            
            # Check validity of configuration-related arguments
            if transfer_function not in transfer_function_types.keys():
                raise ValueError( "'%s' is not a valid transfer_function type. "
                                  "Available options are: %s" \
                                 % (transfer_function, 
                                    transfer_function_types.keys()) )
            if matter_power_spectrum not in matter_power_spectrum_types.keys():
                raise ValueError( "'%s' is not a valid matter_power_spectrum "
                                  "type. Available options are: %s" \
                                 % (matter_power_spectrum, 
                                    matter_power_spectrum_types.keys()) )
            if mass_function not in mass_function_types.keys():
                raise ValueError( "'%s' is not a valid mass_function type. "
                                  "Available options are: %s" \
                                 % (mass_function, 
                                    mass_function_types.keys()) )
            
            # Assign values to new ccl_configuration object
            config = lib.configuration()
            
            config.transfer_function_method = \
                            transfer_function_types[transfer_function]
            config.matter_power_spectrum_method = \
                            matter_power_spectrum_types[matter_power_spectrum]
            config.mass_function_method = \
                            mass_function_types[mass_function]
            
            # Store ccl_configuration for later access
            self.configuration = config
        
        # Create new ccl_cosmology instance
        self.cosmo = lib.cosmology_create(params, config)
        
        # Check status
        if self.cosmo.status != 0:
            raise RuntimeError("(%d): %s" \
                               % (self.cosmo.status, self.cosmo.status_message))
    
    def __del__(self):
        """Free the ccl_cosmology instance that this Cosmology object is managing.

        """
        lib.cosmology_free(self.cosmo)
    
    def __str__(self):
        """Output the cosmological parameters that were set, and their values,
        as well as the status of precomputed quantities and the internal CCL
        status.

        """
        # String of cosmo parameters, from self.params (Parameters object)
        param_str = self.params.__str__()
        
        # String containing precomputation statuses
        precomp_stats = [
            ('has_distances', self.has_distances()),
            ('has_growth',    self.has_growth()),
            ('has_power',     self.has_power()),
            ('has_sigma',     self.has_sigma()),
            ]
        precomp_stat = ["%15s: %s" % stat for stat in precomp_stats]
        precomp_str = "\n".join(precomp_stat)
        
        # String from internal CCL status
        status_str = self.status()
        
        # Return composite string
        string = param_str
        string += "\n\nPrecomputed data\n----------------\n"
        string += precomp_str
        string += "\n\nStatus\n------\n"
        string += status_str
        return string
    
    def __getitem__(self, key):
        """Access cosmological parameter values by name.

        """
        return self.params.__getitem__(key)
    
    def compute_distances(self):
        """Interfaces with src/compute_background.c: ccl_cosmology_compute_distances().
        Sets up the splines for the distances.

        """
        status = 0
        lib.cosmology_compute_distances(self.cosmo, status)
        check(status)
    
    def compute_growth(self):
        """Interfaces with src/ccl_background.c: ccl_cosmology_compute_growth().
        Sets up the splines for the growth function.

        """
        status = 0
        lib.cosmology_compute_growth(self.cosmo, status)
        check(status)
    
    def compute_power(self):
        """Interfaces with src/ccl_power.c: ccl_cosmology_compute_power().
        Sets up the splines for the power spectrum.

        """
        status = 0
        lib.cosmology_compute_power(self.cosmo, status)
        check(status)
    
    def has_distances(self):
        """Checks if the distances have been precomputed.

        Returns:
            True if precomputed, False otherwise.

        """
        return bool(self.cosmo.computed_distances)
    
    def has_growth(self):
        """Checks if the growth function has been precomputed.

        Returns:
            True if precomputed, False otherwise.

        """
        return bool(self.cosmo.computed_growth)
    
    def has_power(self):
        """Checks if the power spectra have been precomputed.

        Returns:
            True if precomputed, False otherwise.

        """
        return bool(self.cosmo.computed_power)
    
    def has_sigma(self):
        """Checks if sigma8 has been computed.

        Returns:
            True if precomputed, False otherwise.

        """
        return bool(self.cosmo.computed_sigma)
    
    def status(self):
        """Get error status of the ccl_cosmology object.

        Note: error status is all currently under development.

        Returns:
            :obj:`str` containing the status message.

        """
        # Get status ID string if one exists
        if self.cosmo.status in error_types.keys():
            status = error_types[self.cosmo.status]
        else:
            status = self.cosmo.status
        
        # Get status message
        msg = self.cosmo.status_message
        
        # Return status information
        return "status(%s): %s" % (status, msg)
        
