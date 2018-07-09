
from pyccl import ccllib as lib
import numpy as np
from warnings import warn
from pyccl.pyutils import check

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
    'class':            lib.boltzmann_class,
}

matter_power_spectrum_types = {
    'halo_model':   lib.halo_model,
    'halomodel':    lib.halo_model,
    'halofit':      lib.halofit,
    'linear':       lib.linear,
    'emu':          lib.emu
}

# List which matter_power_spectrum types are allowed for each transfer_function
valid_transfer_matter_power_combos = {
    'none':             [],
    'emulator':         [lib.emu,],
    'fitting_function': [lib.linear, lib.halofit, lib.halo_model],
    'eisenstein_hu':    [lib.linear, lib.halofit, lib.halo_model],
    'bbks':             [lib.linear, lib.halofit, lib.halo_model],
    'boltzmann':        [lib.linear, lib.halofit],
    'boltzmann_class':  [lib.linear, lib.halofit],
    'class':            [lib.linear, lib.halofit],
    'boltzmann_camb':   [],
    'camb':             [],
}

baryons_power_spectrum_types = {
    'nobaryons':   lib.nobaryons,
    'bcm':      lib.bcm
}

mass_function_types = {
    'angulo':   lib.angulo,
    'tinker':   lib.tinker,
    'tinker10': lib.tinker10,
    'watson':   lib.watson
}

emulator_neutrinos_types = {
	'strict': 	lib.emu_strict,
	'equalize': lib.emu_equalize
}

mnu_types = {
	'list': lib.mnu_list,
	'sum': lib.mnu_sum,
	'sum_inverted': lib.mnu_sum_inverted,
	'sum_equal': lib.mnu_sum_equal, 
}

# Error types
error_types = {
    lib.CCL_ERROR_MEMORY:              'CCL_ERROR_MEMORY',
    lib.CCL_ERROR_LINSPACE:            'CCL_ERROR_LINSPACE',
    lib.CCL_ERROR_INCONSISTENT:        'CCL_ERROR_INCONSISTENT',
    lib.CCL_ERROR_SPLINE:              'CCL_ERROR_SPLINE',
    lib.CCL_ERROR_SPLINE_EV:           'CCL_ERROR_SPLINE_EV',
    lib.CCL_ERROR_INTEG:               'CCL_ERROR_INTEG',
    lib.CCL_ERROR_ROOT:                'CCL_ERROR_ROOT',
    lib.CCL_ERROR_CLASS:               'CCL_ERROR_CLASS',
    lib.CCL_ERROR_COMPUTECHI:          'CCL_ERROR_COMPUTECHI',
    lib.CCL_ERROR_MF:                  'CCL_ERROR_MF',
    lib.CCL_ERROR_HMF_INTERP:          'CCL_ERROR_HMF_INTERP',
    lib.CCL_ERROR_PARAMETERS:          'CCL_ERROR_PARAMETERS',
    lib.CCL_ERROR_NU_INT:	           'CCL_ERROR_NU_INT',
    lib.CCL_ERROR_EMULATOR_BOUND:      'CCL_ERROR_EMULATOR_BOUND',
    lib.CCL_ERROR_MISSING_CONFIG_FILE: 'CCL_ERROR_MISSING_CONFIG_FILE',
}

class Parameters(object):
    """The Parameters class contains cosmological parameters.

    """
    
    def __init__(self, Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
                 Omega_k=0., Neff = 3.046, m_nu=0., mnu_type = None, w0=-1., 
                 wa=0., bcm_log10Mc=np.log10(1.2e14), bcm_etab=0.5, bcm_ks=55., 
                 sigma8=None, z_mg=None, df_mg=None):
        """
        Creates a set of cosmological parameters.

        Note:
            Although some arguments default to `None`, they will raise a 
            ValueError inside this function if not specified, so they are not 
            optional.
        
        Args:
            Omega_c (float): Cold dark matter density fraction.
            Omega_b (float): Baryonic matter density fraction.
            h (float): Hubble constant divided by 100 km/s/Mpc; unitless.
            A_s (float): Power spectrum normalization. Optional if sigma8 is 
                         specified.
            n_s (float): Primordial scalar perturbation spectral index.
            Omega_k (float, optional): Curvature density fraction. Defaults to 0.
            Neff (float, optional): Effective number of neutrino species. 
                                    Defaults to 3.046
            m_nu (float or array-like, optional): If float: total mass in eV of 
            the massive neutrinos present. If array-like, masses of 3 neutrino
            species (must have length 3).
			mnu_type (string): treatment for neutrinos. 
			        Available: 'sum', 'sum_inverted', 'sum_equal', 'list'. 
			        Default if m_nu is a float is 'sum', default if m_nu is 
			        array-like with length 3 is 'list'.
            w0 (float, optional): First order term of dark energy equation of 
                                  state. Defaults to -1.
            wa (float, optional): Second order term of dark energy equation of 
                                  state. Defaults to 0.
            bcm_log10Mc (float, optional): One of the parameters of the BCM model.
            bcm_etab (float, optional): One of the parameters of the BCM model.
            bcm_ks (float, optional): One of the parameters of the BCM model.
            sigma8 (float): Variance of matter density perturbations at 8 Mpc/h
                            scale. Optional if A_s is specified.
            df_mg (:obj: array_like): Perturbations to the GR growth rate as a 
                                      function of redshift, Delta f. Used to 
                                      implement simple modified growth 
                                      scenarios.
            z_mg (:obj: array_like): Array of redshifts corresponding to df_mg.

        """
        # Set current ccl_parameters object to None
        self.parameters = None
        
         # Set nz_mg (no. of redshift bins for modified growth fns.)
        if z_mg is not None and df_mg is not None:
            # Get growth array size and do sanity check
            z_mg = np.atleast_1d(z_mg)
            df_mg = np.atleast_1d(df_mg)
            assert z_mg.size == df_mg.size
            nz_mg = z_mg.size
        else:
            # If one or both of the MG growth arrays are set to zero, disable 
            # all of them
            if z_mg is not None or df_mg is not None:
                raise ValueError("Must specify both z_mg and df_mg.")
            z_mg = None
            df_mg = None
            nz_mg = -1
        
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
        
        if isinstance(m_nu, float):
            if mnu_type == None: mnu_type = 'sum'
            m_nu = [m_nu]
        elif hasattr(m_nu, "__len__"):
            if (len(m_nu) != 3):
                raise ValueError("m_nu must be a float or array-like object "
                                 "with length 3.")
            elif ((mnu_type=='sum') \
               or (mnu_type=='sum_inverted') \
               or (mnu_type=='sum_equal')):
                raise ValueError("mnu type '%s' cannot be passed with a list "
                                 "of neutrino masses, only with a sum." \
                                 % mnu_type)
            elif (mnu_type==None):
                mnu_type = 'list'  # False
        else:
            raise ValueError("m_nu must be a float or array-like object with "
                             "length 3.")
        
        
        # Check if any compulsory parameters are not set
        compul = [Omega_c, Omega_b, Omega_k, w0, wa, h, norm_pk, n_s]
        names = ['Omega_c', 'Omega_b', 'Omega_k', 'w0', 'wa', 
                 'h', 'norm_pk', 'n_s']

        for nm, item in zip(names, compul):
            if item is None:
                raise ValueError("Necessary parameter '%s' was not set "
                                 "(or set to None)." % nm)                     
                                 
        # Create new instance of ccl_parameters object
        status = 0 # Create an internal status variable; needed to check massive neutrino integral.
        if (nz_mg== -1):
            # Create ccl_parameters without modified growth
            self.parameters, status \
            = lib.parameters_create_nu( Omega_c, Omega_b, Omega_k, Neff, 
                                             w0, wa, h, norm_pk, 
                                             n_s, bcm_log10Mc, bcm_etab, bcm_ks, 
                                             mnu_types[mnu_type], m_nu, status ) 
                                             
        else:
            # Create ccl_parameters with modified growth arrays
            self.parameters, status \
            = lib.parameters_create_nu_vec( Omega_c, Omega_b, Omega_k, Neff, 
                                             w0, wa, h, norm_pk, 
                                             n_s, bcm_log10Mc, bcm_etab, bcm_ks, 
                                             z_mg, df_mg, mnu_types[mnu_type], m_nu, status )
        check(status)    
    
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
        """
        try:
            # First check if the key already exists (otherwise the parameter 
            # would be silently added to the ccl_parameters class instance)
            getattr(self.parameters, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        
        # Set value of parameter
        setattr(self.parameters, key, val)
        # TODO: Should update/replace CCL objects appropriately
        """
    
    def __del__(self):
        """
        Free the ccl_parameters instance that this Parameters object is 
        managing.
        """
        if hasattr(self, 'parameters'):
            if self.parameters is not None: 
                lib.parameters_free(self.parameters)
    
    
    def __str__(self):
        """
        Output the parameters that were set, and their values.
        """
        params = ['Omega_c', 'Omega_b', 'Omega_m', 'Omega_k', 'Omega_l',
                  'w0', 'wa', 'H0', 'h', 'A_s', 'n_s', 'bcm_log10Mc', 
                  'bcm_etab', 'bcm_ks',
                  'Neff', 'mnu', 'Omega_n_mass', 'Omega_n_rel',
                  'T_CMB', 'Omega_g', 'z_star', 'has_mgrowth']
        
        # Get values of parameters
        vals = []
        for p in params:
            try:
                v = getattr(self.parameters, p)
            except:
                # Parameter name was not found in ccl_parameters struct
                v = "Not found"
            vals.append( "%15s: %s" % (p, v) )
        string = "Parameters\n----------\n"
        string += "\n".join(vals)
        return string


class Cosmology(object):
    """Wrapper for the ccl_cosmology object.

    Includes cosmological parameters and cached data.

    """
    
    def __init__(self, 
                 params=None, config=None,
                 Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
                 Omega_k=0., Neff=3.046, m_nu=0., mnu_type = None, w0=-1., wa=0.,
                 bcm_log10Mc=np.log10(1.2e14), bcm_etab=0.5, bcm_ks=55., 
                 sigma8=None, z_mg=None, df_mg=None, 
                 transfer_function='boltzmann_class',
                 matter_power_spectrum='halofit',
                 baryons_power_spectrum='nobaryons',
                 mass_function='tinker10', emulator_neutrinos='strict'):
        """Creates a wrapper for ccl_cosmology.

        Args:
            params (:obj:`Parameters`): Cosmological parameters object.
            config (:obj:`ccl_configuration`, optional): Configuration for how 
            to use CCL. Takes precident over any other passed in configuration. 
            Defaults to None.
            transfer_function (:obj:`str`, optional): The transfer function to 
            use. Defaults to `boltzmann_class`.
            matter_power_spectrum (:obj:`str`, optional): The matter power 
            spectrum to use. Defaults to `halofit`.
            baryons_power_spectrum (:obj:`str`, optional): The correction from 
            baryonic effects to be implemented. Defaults to `nobaryons`.
            mass_function (:obj:`str`, optional): The mass function to use. 
            Defaults to `tinker` (2010).
            emulator_neutrinos: `str`, optional): If using the emulator for 
            the power spectrum, specified treatment of unequal neutrinos.
            Options are 'strict', which will raise an error and quit if the 
            user fails to pass either a set of three equal masses or a sum with 
            mnu_type = 'equal', and 'equalize', which will redistribute masses
            to be equal right before calling the emualtor but results in
            internal inconsistencies. Defaults to `strict`.

        """
        
        # Use either input cosmology parameters or Parameters() object
        if params is None:
            # Create new Parameters object
            params = Parameters(Omega_c=Omega_c, Omega_b=Omega_b, h=h, A_s=A_s, 
                                n_s=n_s, Omega_k=Omega_k, Neff = Neff, 
                                m_nu=m_nu, mnu_type=mnu_type, w0=w0, wa=wa, 
                                sigma8=sigma8, bcm_log10Mc=bcm_log10Mc, 
                                bcm_etab=bcm_etab, bcm_ks=bcm_ks, 
                                z_mg=z_mg, df_mg=df_mg)

            self.params = params
            params = params.parameters # We only need the ccl_parameters object
        elif isinstance(params, lib.parameters):
            # Raise an error if ccl_parameters given directly
            raise TypeError("Must pass a Parameters() object, not ccl_parameters.")
        elif isinstance(params, Parameters):
            # Parameters object given directly
            self.params = params
            
            # Warn if any cosmological parameters were specified at the same 
            # time as a Parameters() object; they will be ignored
            argtest = [Omega_c==None, Omega_b==None, h==None, A_s==None, 
                       n_s==None, Omega_k==0., Neff==3.046, m_nu==0., 
                       mnu_type==None, w0==-1., wa==0., 
                       bcm_log10Mc==np.log10(1.2e14), bcm_etab==0.5, 
                       bcm_ks==55., sigma8==None, z_mg==None, df_mg==None]
            
            if not all(arg == True for arg in argtest):
                warn("Cosmological parameter kwargs are ignored if 'params' is "
                     "not None", UserWarning)
        else:
            raise TypeError("'params' is not a valid Parameters object.")
        
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
                raise KeyError( "'%s' is not a valid transfer_function type. "
                                  "Available options are: %s" \
                                 % (transfer_function, 
                                    transfer_function_types.keys()) )
            if matter_power_spectrum not in matter_power_spectrum_types.keys():
                raise KeyError( "'%s' is not a valid matter_power_spectrum "
                                  "type. Available options are: %s" \
                                 % (matter_power_spectrum, 
                                    matter_power_spectrum_types.keys()) )
            if baryons_power_spectrum not in baryons_power_spectrum_types.keys():
                raise KeyError( "'%s' is not a valid baryons_power_spectrum "
                                  "type. Available options are: %s" \
                                 % (baryons_power_spectrum, 
                                    baryons_power_spectrum_types.keys()) )
            if mass_function not in mass_function_types.keys():
                raise KeyError( "'%s' is not a valid mass_function type. "
                                  "Available options are: %s" \
                                 % (mass_function, 
                                    mass_function_types.keys()) )
            if emulator_neutrinos not in emulator_neutrinos_types.keys():
                raise ValueError( "'%s' is not a valid emulator neutrinos method. "
                                  "Available options are: %s" \
                                 % (emulator_neutrinos, 
                                    emulator_neutrinos_types.keys()) )
            
            # Check for valid transfer fn/matter power spectrum combination
            if matter_power_spectrum_types[matter_power_spectrum] \
              not in valid_transfer_matter_power_combos[transfer_function]:
                raise ValueError("matter_power_spectrum '%s' can't be used "
                                 "with transfer_function '%s'." \
                                 % (matter_power_spectrum, transfer_function))
            
            # Assign values to new ccl_configuration object
            config = lib.configuration()
            
            config.transfer_function_method = \
                            transfer_function_types[transfer_function]
            config.matter_power_spectrum_method = \
                            matter_power_spectrum_types[matter_power_spectrum]
            config.baryons_power_spectrum_method = \
                            baryons_power_spectrum_types[baryons_power_spectrum]
            config.mass_function_method = \
                            mass_function_types[mass_function]
            config.emulator_neutrinos_method = \
                            emulator_neutrinos_types[emulator_neutrinos]
            
            # Store ccl_configuration for later access
            self.configuration = config
        
        # Create new ccl_cosmology instance
        self.cosmo = lib.cosmology_create(self.params.parameters, config)
        
        # Check status
        if self.cosmo.status != 0:
            raise RuntimeError("(%d): %s" \
                               % (self.cosmo.status, self.cosmo.status_message))
    
    def __del__(self):
        """Free the ccl_cosmology instance that this Cosmology object is managing.

        """
        if hasattr(self, "cosmo"):
            if self.cosmo is not None:
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
        status = lib.cosmology_compute_distances(self.cosmo, status)
        check(status, self.cosmo)
    
    def compute_growth(self):
        """Interfaces with src/ccl_background.c: ccl_cosmology_compute_growth().
        Sets up the splines for the growth function.

        """
        status = 0
        status = lib.cosmology_compute_growth(self.cosmo, status)
        check(status, self.cosmo)
    
    def compute_power(self):
        """Interfaces with src/ccl_power.c: ccl_cosmology_compute_power().
        Sets up the splines for the power spectrum.

        """
        status = 0
        status = lib.cosmology_compute_power(self.cosmo, status)
        check(status, self.cosmo)
    
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
        
