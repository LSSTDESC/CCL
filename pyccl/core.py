"""The core functionality of ccl, including the core data types. This includes
the cosmology and parameters objects used to instantiate a model from which one
can compute a set of theoretical predictions.

The classes in this module accept strings indicating which model to use
for various physical quantities (e.g., the transfer function). The various
options are as follows.

transfer_function options
  - 'emulator': the transfer function defined by the Comsic Emu
  - 'fitting_function': the Eisenstein and Hu (1998) fitting function
  - 'eisenstein_hu': the Eisenstein and Hu (1998) fitting function
  - 'bbks': the BBKS approximation
  - 'boltzmann': use CLASS to compute the transfer function
  - 'boltzmann_class': use CLASS to compute the transfer function
  - 'class': use CLASS to compute the transfer function
  - 'boltzmann_camb': not implemented
  - 'camb': not implemented

matter_power_spectrum options
  - 'halo_model': use a halo model
  - 'halofit': use HALOFIT
  - 'linear': neglect non-linear power spectrum contributions
  - 'emu': use the Cosmic Emu

baryons_power_spectrum options
  - 'nobaryons': neglect baryonic contributions to the power spectrum
  - 'bcm': use the baryonic correction model

mass_function options
  - 'tinker': the Tinker et al. (2008) mass function
  - 'tinker10': the Tinker et al. (2010) mass function
  - 'watson': the Watson et al. mass function
  - 'angulo': the Angulo et al. mass function
  - 'shethtormen': the Sheth and Tormen mass function

halo_concentration options
  - 'bhattacharya2011': Bhattacharya et al. (2011) relation
  - 'duffy2008': Duffy et al. (2008) relation
  - 'constant_concentration': use a constant concentration

mnu_type options
  This parameter specifies the model for massive
  neutrinos.
    - 'list': specify each mass yourself in eV
    - 'sum': use the normal hierarchy to convert total mass to individual
      masses
    - 'sum_inverted': use the inverted hierarchy to convert total mass to
      individual masses
    - 'sum_equal': assume equal masses when converting the total mass to
      individual masses

emulator_neutrinos options
  This parameter specifies how to handle inconsistencies in the treatment of
  neutrinos between the Cosmic Emu (equal masses) and other models.
    - 'strict': fail unless things are absolutely consistent
    - 'equalize': redistribute the total mass equaly before using the Cosmic
      Emu. This option may result in slight internal inconsistencies in the
      physical model assumed for neutrinos.
"""
import numpy as np
import yaml

from . import ccllib as lib
from .errors import CCLError

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
    'halofit':      lib.halofit,
    'linear':       lib.linear,
    'emu':          lib.emu
}

# List which matter_power_spectrum types are allowed for each transfer_function
valid_transfer_matter_power_combos = {
    'none':             [],
    'emulator':         [lib.emu, ],
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
    'bcm':         lib.bcm
}

mass_function_types = {
    'angulo':      lib.angulo,
    'tinker':      lib.tinker,
    'tinker10':    lib.tinker10,
    'watson':      lib.watson,
    'shethtormen': lib.shethtormen
}

halo_concentration_types = {
    'bhattacharya2011':          lib.bhattacharya2011,
    'duffy2008':                 lib.duffy2008,
    'constant_concentration':    lib.constant_concentration,
}

emulator_neutrinos_types = {
    'strict': lib.emu_strict,
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


class Cosmology(object):
    """A cosmology including parameters and associated data.

    .. note:: Although some arguments default to `None`, they will raise a
              ValueError inside this function if not specified, so they are not
              optional.

    .. note:: The parameter Omega_g can be used to set the radiation density
              (not including relativistic neutrinos) to zero. Doing this will
              give you a model that is physically inconsistent since the
              temperature of the CMB will still be non-zero. Note however
              that this approximation is common for late-time LSS computations.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    Args:
        Omega_c (:obj:`float`): Cold dark matter density fraction.
        Omega_b (:obj:`float`): Baryonic matter density fraction.
        h (:obj:`float`): Hubble constant divided by 100 km/s/Mpc; unitless.
        A_s (:obj:`float`): Power spectrum normalization. Exactly one of A_s
            and sigma_8 is required.
        sigma8 (:obj:`float`): Variance of matter density perturbations at
            an 8 Mpc/h scale. Exactly one of A_s and sigma_8 is required.
        n_s (:obj:`float`): Primordial scalar perturbation spectral index.
        Omega_k (:obj:`float`, optional): Curvature density fraction.
            Defaults to 0.
        Omega_g (:obj:`float`, optional): Density in relativistic species
            except massless neutrinos. The default of `None` corresponds
            to setting this from the CMB temperature. Note that if a non-`None`
            value is given, this may result in a physically inconsistent model
            because the CMB temperature will still be non-zero in the
            parameters.
        Neff (:obj:`float`, optional): Effective number of massless
            neutrinos present. Defaults to 3.046.
        m_nu (:obj:`float`, optional): Total mass in eV of the massive
            neutrinos present. Defaults to 0.
        mnu_type (:obj:`str`, optional): The type of massive neutrinos.
        w0 (:obj:`float`, optional): First order term of dark energy equation
            of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy equation
            of state. Defaults to 0.
        bcm_log10Mc (:obj:`float`, optional): One of the parameters of the
            BCM model. Defaults to `np.log10(1.2e14)`.
        bcm_etab (:obj:`float`, optional): One of the parameters of the BCM
            model. Defaults to 0.5.
        bcm_ks (:obj:`float`, optional): One of the parameters of the BCM
            model. Defaults to 55.0.
        df_mg (array_like, optional): Perturbations to the GR growth rate as
            a function of redshift :math:`\\Delta f`. Used to implement simple
            modified growth scenarios.
        z_mg (array_like, optional): Array of redshifts corresponding to df_mg.
        transfer_function (:obj:`str`, optional): The transfer function to
            use. Defaults to 'boltzmann_class'.
        matter_power_spectrum (:obj:`str`, optional): The matter power
            spectrum to use. Defaults to 'halofit'.
        baryons_power_spectrum (:obj:`str`, optional): The correction from
            baryonic effects to be implemented. Defaults to 'nobaryons'.
        mass_function (:obj:`str`, optional): The mass function to use.
            Defaults to 'tinker10' (2010).
        halo_concentration (:obj:`str`, optional): The halo concentration
            relation to use. Defaults to Duffy et al. (2008) 'duffy2008'.
        emulator_neutrinos: `str`, optional): If using the emulator for
            the power spectrum, specified treatment of unequal neutrinos.
            Options are 'strict', which will raise an error and quit if the
            user fails to pass either a set of three equal masses or a sum with
            mnu_type = 'sum_equal', and 'equalize', which will redistribute
            masses to be equal right before calling the emualtor but results in
            internal inconsistencies. Defaults to 'strict'.
    """
    def __init__(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None,
            Omega_k=0., Omega_g=None, Neff=3.046, m_nu=0., mnu_type=None,
            w0=-1., wa=0., bcm_log10Mc=np.log10(1.2e14), bcm_etab=0.5,
            bcm_ks=55., z_mg=None, df_mg=None,
            transfer_function='boltzmann_class',
            matter_power_spectrum='halofit',
            baryons_power_spectrum='nobaryons',
            mass_function='tinker10',
            halo_concentration='duffy2008',
            emulator_neutrinos='strict'):

        # going to save these for later
        self._params_init_kwargs = dict(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, Omega_g=Omega_g, Neff=Neff, m_nu=m_nu,
            mnu_type=mnu_type, w0=w0, wa=wa, bcm_log10Mc=bcm_log10Mc,
            bcm_etab=bcm_etab, bcm_ks=bcm_ks, z_mg=z_mg, df_mg=df_mg)

        self._config_init_kwargs = dict(
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum,
            baryons_power_spectrum=baryons_power_spectrum,
            mass_function=mass_function,
            halo_concentration=halo_concentration,
            emulator_neutrinos=emulator_neutrinos)

        self._build_cosmo()

    def _build_cosmo(self):
        """Assemble all of the input data into a valid ccl_cosmology object."""
        # We have to make all of the C stuff that goes into a cosmology
        # and then we make the cosmology.
        self._build_parameters(**self._params_init_kwargs)
        self._build_config(**self._config_init_kwargs)
        self.cosmo = lib.cosmology_create(self._params, self._config)

        if self.cosmo.status != 0:
            raise CCLError(
                "(%d): %s"
                % (self.cosmo.status, self.cosmo.status_message))

    def write_yaml(self, filename):
        """Write a YAML representation of the parameters to file.

        Args:
            filename (:obj:`str`) Filename to write parameters to.
        """
        # NOTE: we use the C yaml dump here so that the parameters
        # dumped by this object are compatible with the C yaml load function.
        status = 0
        lib.parameters_write_yaml(self._params, filename, status)

        # Check status
        if status != 0:
            raise IOError("Unable to write YAML file {}".format(filename))

    @classmethod
    def read_yaml(cls, filename):
        """Read the parameters from a YAML file.

        Args:
            filename (:obj:`str`) Filename to read parameters from.
        """
        with open(filename, 'r') as fp:
            params = yaml.load(fp)

        # Now we assemble an init for the object since the CCL YAML has
        # extra info we don't need and different formatting.
        inits = dict(
            Omega_c=params['Omega_c'],
            Omega_b=params['Omega_b'],
            h=params['h'],
            n_s=params['n_s'],
            sigma8=None if params['sigma8'] == 'nan' else params['sigma8'],
            A_s=None if params['A_s'] == 'nan' else params['A_s'],
            Omega_k=params['Omega_k'],
            Neff=params['Neff'],
            w0=params['w0'],
            wa=params['wa'],
            bcm_log10Mc=params['bcm_log10Mc'],
            bcm_etab=params['bcm_etab'],
            bcm_ks=params['bcm_ks'])
        if 'z_mg' in params:
            inits['z_mg'] = params['z_mg']
            inits['df_mg'] = params['df_mg']

        if 'mnu' in params:
            inits['m_nu'] = params['mnu']
            inits['mnu_type'] = 'list'

        return cls(**inits)

    def _build_config(
            self, transfer_function=None, matter_power_spectrum=None,
            baryons_power_spectrum=None,
            mass_function=None, halo_concentration=None,
            emulator_neutrinos=None):
        """Build a ccl_configuration struct.

        This function builds C ccl_configuration struct. This structure
        controls which various approximations are used for the transfer
        function, matter power spectrum, baryonic effect in the matter
        power spectrum, mass function, halo concentration relation, and
        neutrino effects in the emulator.

        It also does some error checking on the inputs to make sure they
        are valid and physically consistent.
        """

        # Check validity of configuration-related arguments
        if transfer_function not in transfer_function_types.keys():
            raise ValueError(
                "'%s' is not a valid transfer_function type. "
                "Available options are: %s"
                % (transfer_function,
                   transfer_function_types.keys()))
        if matter_power_spectrum not in matter_power_spectrum_types.keys():
            raise ValueError(
                "'%s' is not a valid matter_power_spectrum "
                "type. Available options are: %s"
                % (matter_power_spectrum,
                   matter_power_spectrum_types.keys()))
        if (baryons_power_spectrum not in
                baryons_power_spectrum_types.keys()):
            raise ValueError(
                "'%s' is not a valid baryons_power_spectrum "
                "type. Available options are: %s"
                % (baryons_power_spectrum,
                   baryons_power_spectrum_types.keys()))
        if mass_function not in mass_function_types.keys():
            raise ValueError(
                "'%s' is not a valid mass_function type. "
                "Available options are: %s"
                % (mass_function,
                   mass_function_types.keys()))
        if halo_concentration not in halo_concentration_types.keys():
            raise ValueError(
                "'%s' is not a valid halo_concentration type. "
                "Available options are: %s"
                % (halo_concentration,
                   halo_concentration_types.keys()))
        if emulator_neutrinos not in emulator_neutrinos_types.keys():
            raise ValueError("'%s' is not a valid emulator neutrinos "
                             "method. Available options are: %s"
                             % (emulator_neutrinos,
                                emulator_neutrinos_types.keys()))

        # Check for valid transfer fn/matter power spectrum combination
        if (matter_power_spectrum_types[matter_power_spectrum]
                not in
                valid_transfer_matter_power_combos[transfer_function]):
            raise ValueError("matter_power_spectrum '%s' can't be used "
                             "with transfer_function '%s'."
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
        config.halo_concentration_method = \
            halo_concentration_types[halo_concentration]
        config.emulator_neutrinos_method = \
            emulator_neutrinos_types[emulator_neutrinos]

        # Store ccl_configuration for later access
        self._config = config

    def _build_parameters(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None, sigma8=None,
            A_s=None, Omega_k=None, Neff=None, m_nu=None, mnu_type=None,
            w0=None, wa=None, bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            z_mg=None, df_mg=None, Omega_g=None):
        """Build a ccl_parameters struct"""

        # Set nz_mg (no. of redshift bins for modified growth fns.)
        if z_mg is not None and df_mg is not None:
            # Get growth array size and do sanity check
            z_mg = np.atleast_1d(z_mg)
            df_mg = np.atleast_1d(df_mg)
            if z_mg.size != df_mg.size:
                raise ValueError(
                    "The parameters `z_mg` and `dF_mg` are "
                    "not the same shape!")
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
        if ((A_s is None and sigma8 is None) or
                (A_s is not None and sigma8 is not None)):
            raise ValueError("Must set either A_s or sigma8 and not both.")

        # Set norm_pk to either A_s or sigma8
        norm_pk = A_s if A_s is not None else sigma8

        # The C library decides whether A_s or sigma8 was the input parameter
        # based on value, so we need to make sure this is consistent too
        if norm_pk >= 1e-5 and A_s is not None:
            raise ValueError("A_s must be less than 1e-5.")

        if norm_pk < 1e-5 and sigma8 is not None:
            raise ValueError("sigma8 must be greater than 1e-5.")

        if isinstance(m_nu, float):
            if mnu_type is None:
                mnu_type = 'sum'
            m_nu = [m_nu]
        elif hasattr(m_nu, "__len__"):
            if (len(m_nu) != 3):
                raise ValueError("m_nu must be a float or array-like object "
                                 "with length 3.")
            elif ((mnu_type == 'sum') or
                    (mnu_type == 'sum_inverted') or
                    (mnu_type == 'sum_equal')):
                raise ValueError(
                    "mnu type '%s' cannot be passed with a list "
                    "of neutrino masses, only with a sum." % mnu_type)
            elif mnu_type is None:
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
        # Create an internal status variable; needed to check massive neutrino
        # integral.
        status = 0
        if nz_mg == -1:
            # Create ccl_parameters without modified growth
            self._params, status = lib.parameters_create_nu(
                Omega_c, Omega_b, Omega_k, Neff,
                w0, wa, h, norm_pk,
                n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                mnu_types[mnu_type], m_nu, status)
        else:
            # Create ccl_parameters with modified growth arrays
            self._params, status = lib.parameters_create_nu_vec(
                Omega_c, Omega_b, Omega_k, Neff,
                w0, wa, h, norm_pk,
                n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                z_mg, df_mg, mnu_types[mnu_type], m_nu, status)
        check(status)

        # we cannot set omega_g via the C code directly. Thus we set it by hand
        # and then put any difference into omega_l, which follows the
        # what the C code does.
        if Omega_g is not None:
            total = self._params.Omega_g + self._params.Omega_l
            self._params.Omega_g = Omega_g
            self._params.Omega_l = total - Omega_g

    def __getitem__(self, key):
        """Access parameter values by name."""
        try:
            val = getattr(self._params, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        return val

    def __setitem__(self, key, val):
        """Set parameter values by name."""
        raise NotImplementedError("Cosmology objects are immutable; create a "
                                  "new Cosmology() instance instead.")

    def __del__(self):
        """Free the C memory this object is managing as it is being garbage
        collected (hopefully)."""
        if hasattr(self, "cosmo"):
            if self.cosmo is not None:
                lib.cosmology_free(self.cosmo)
        if hasattr(self, "_params"):
            if self._params is not None:
                lib.parameters_free(self._params)

    def __getstate__(self):
        # we are removing any C data before pickling so that the
        # is pure python when pickled.
        state = self.__dict__.copy()
        state.pop('cosmo', None)
        state.pop('_params', None)
        state.pop('_config', None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # we removed the C data when it was pickled, so now we unpickle
        # and rebuild the C data
        self._build_cosmo()

    def __repr__(self):
        """Make an eval-able string.

        This feature can be used like this:

        >>> import pyccl
        >>> cosmo = pyccl.Cosmology(...)
        >>> cosmo2 = eval(repr(cosmo))
        """

        string = "pyccl.Cosmology("
        string += ", ".join(
            "%s=%s" % (k, v)
            for k, v in self._params_init_kwargs.items()
            if k not in ['m_nu', 'mnu_type', 'z_mg', 'df_mg'])

        if hasattr(self._params_init_kwargs['m_nu'], '__len__'):
            string += ", m_nu=[%s, %s, %s]" % tuple(
                self._params_init_kwargs['m_nu'])
        else:
            string += ', m_nu=%s' % self._params_init_kwargs['m_nu']

        if self._params_init_kwargs['mnu_type'] is not None:
            string += ", mnu_type='%s'" % self._params_init_kwargs['mnu_type']
        else:
            string += ', mnu_type=None'

        if self._params_init_kwargs['z_mg'] is not None:
            vals = ", ".join(
                ["%s" % v for v in self._params_init_kwargs['z_mg']])
            string += ", z_mg=[%s]" % vals
        else:
            string += ", z_mg=%s" % self._params_init_kwargs['z_mg']

        if self._params_init_kwargs['df_mg'] is not None:
            vals = ", ".join(
                ["%s" % v for v in self._params_init_kwargs['df_mg']])
            string += ", df_mg=[%s]" % vals
        else:
            string += ", df_mg=%s" % self._params_init_kwargs['df_mg']

        string += ", "
        string += ", ".join(
            "%s='%s'" % (k, v) for k, v in self._config_init_kwargs.items())
        string += ")"

        return string

    def compute_distances(self):
        """Compute the distance splines."""
        status = 0
        status = lib.cosmology_compute_distances(self.cosmo, status)
        check(status, self.cosmo)

    def compute_growth(self):
        """Compute the growth function."""
        status = 0
        status = lib.cosmology_compute_growth(self.cosmo, status)
        check(status, self.cosmo)

    def compute_power(self):
        """Compute the power spectrum."""
        status = 0
        status = lib.cosmology_compute_power(self.cosmo, status)
        check(status, self.cosmo)

    def has_distances(self):
        """Checks if the distances have been precomputed.

        Returns:
            bool: True if precomputed, False otherwise.
        """
        return bool(self.cosmo.computed_distances)

    def has_growth(self):
        """Checks if the growth function has been precomputed.

        Returns:
            bool: True if precomputed, False otherwise.
        """
        return bool(self.cosmo.computed_growth)

    def has_power(self):
        """Checks if the power spectra have been precomputed.

        Returns:
            bool: True if precomputed, False otherwise.
        """
        return bool(self.cosmo.computed_power)

    def has_sigma(self):
        """Checks if sigma8 has been computed.

        Returns:
            bool: True if precomputed, False otherwise.
        """
        return bool(self.cosmo.computed_sigma)

    def status(self):
        """Get error status of the ccl_cosmology object.

        .. note:: error statuses are currently under development.

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


def check(status, cosmo=None):
    """Check the status returned by a ccllib function.

    Args:
        status (int or :obj:`core.error_types`): Flag or error describing the
                                                 success of a function.
    """
    # Check for normal status (no action required)
    if status == 0:
        return

    # Get status message from Cosmology object, if there is one
    if cosmo is not None:
        msg = cosmo.cosmo.status_message
    else:
        msg = ""

    # Check for known error status
    if status in error_types.keys():
        raise CCLError("Error %s: %s" % (error_types[status], msg))

    # Check for unknown error
    if status != 0:
        raise CCLError("Error %d: %s" % (status, msg))
