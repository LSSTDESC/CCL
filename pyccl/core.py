
import ccllib as lib

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
    
    def __init__(self, Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
                 Omega_k=0., Omega_n=0., w0=-1., wa=0.,
                 zarr_mgrowth=None, dfarr_mgrowth=None):
        """
        Class containing a set of cosmological parameters.
        """
        # Set current ccl_parameters object to None
        self.parameters = None
        
        # Set nz_mgrowth (no. of redshift bins for modified growth fns.)
        if zarr_mgrowth is not None and dfarr_mgrowth is not None:
            # Get growth array size and do sanity check
            assert zarr_mgrowth.size == dfarr_mgrowth.size
            nz_mgrowth = zarr_mgrowth.size
        else:
            # If one or both of the MG growth arrays are set to zero, disable 
            # all of them
            zarr_mgrowth = None
            dfarr_mgrowth = None
            nz_mgrowth = -1
        
        # Check if any compulsory parameters are not set
        compul = [Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, A_s, n_s]
        names = ['Omega_c', 'Omega_b', 'Omega_k', 'Omega_n', 'w0', 'wa', 
                 'h', 'A_s', 'n_s']
        for nm, item in zip(names, compul):
            if item is None:
                raise ValueError("Necessary parameter '%s' was not set "
                                 "(or set to None)." % nm)
        
        # Create new instance of ccl_parameters object
        self.parameters = lib.parameters_create(
                                Omega_c, Omega_b, Omega_k, Omega_n, 
                                w0, wa, h, A_s, n_s, 
                                nz_mgrowth, zarr_mgrowth, dfarr_mgrowth)
    
    def __getitem__(self, key):
        """
        Access parameter values by name.
        """
        try:
            val = getattr(self.parameters, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        return val
    
    def __setitem__(self, key, val):
        """
        Set parameter values by name.
        """
        try:
            # First check if the key already exists (otherwise the parameter 
            # would be silently added to the ccl_parameters class instance)
            getattr(self.parameters, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        
        # Set value
        setattr(self.parameters, key, val)
        # FIXME: Should trigger update process in CCL to ensure all stored data 
        # are consistent
    
    def __str__(self):
        """
        Output the parameters that were set, and their values.
        """
        params = ['Omega_c', 'Omega_b', 'Omega_m', 'Omega_n', 'Omega_k', 
                  'w0', 'wa', 'H0', 'h', 'A_s', 'n_s', 'Omega_g', 'T_CMB', 
                  'sigma_8', 'Omega_l', 'z_star', 'has_mgrowth']
  
        vals = ["%15s: %s" % (p, getattr(self.parameters, p)) for p in params]
        string = "Parameters\n----------\n"
        string += "\n".join(vals)
        return string
        


class Cosmology(object):
    
    def __init__(self, params, config=None, 
                 transfer_function='boltzmann_class',
                 matter_power_spectrum='halofit',
                 mass_function='tinker'):
        """
        Class containing a ccl_cosmology object, including cosmological 
        parameters and cached data.
        """
        # Check the type of the input params object
        if isinstance(params, lib.parameters):
            pass
        elif isinstance(params, Parameters):
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
        """
        Free the ccl_cosmology instance that this Cosmology object is managing.
        """
        lib.cosmology_free(self.cosmo)
    
    def compute_distances(self):
        lib.cosmology_compute_distances(self.cosmo)
    
    def compute_growth(self):
        lib.cosmology_compute_growth(self.cosmo)
    
    def compute_power(self):
        lib.cosmology_compute_power(self.cosmo)
    
    # Check which data have been precomputed
    def has_distances(self):
        return bool(self.cosmo.computed_distances)
    
    def has_growth(self):
        return bool(self.cosmo.computed_growth)
    
    def has_power(self):
        return bool(self.cosmo.computed_power)
    
    def has_sigma(self):
        return bool(self.cosmo.computed_sigma)
    
    # Return status (for error checking)
    def status(self):
        # Get status string if one exists
        if self.cosmo.status in error_types.keys():
            status = error_types[self.cosmo.status]
        else:
            status = self.cosmo.status
        
        return "status(%s): %s" % (status, self.cosmo.status_message)
        
