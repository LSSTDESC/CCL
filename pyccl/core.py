"""The core functionality of ccl, including the core data types. This includes
the cosmology and parameters objects used to instantiate a model from which one
can compute a set of theoretical predictions.
"""
import warnings
import numpy as np
import yaml

from . import ccllib as lib
from .errors import CCLError, CCLWarning
from ._types import error_types
from .boltzmann import get_class_pk_lin, get_camb_pk_lin, get_isitgr_pk_lin
from .pyutils import check
from .pk2d import Pk2D
from .bcm import bcm_correct_pk2d

# Configuration types
transfer_function_types = {
    None: lib.transfer_none,
    'eisenstein_hu': lib.eisenstein_hu,
    'bbks': lib.bbks,
    'boltzmann_class': lib.boltzmann_class,
    'boltzmann_camb': lib.boltzmann_camb,
    'boltzmann_isitgr': lib.boltzmann_isitgr,
    'calculator': lib.pklin_from_input
}

matter_power_spectrum_types = {
    None: lib.pknl_none,
    'halo_model': lib.halo_model,
    'halofit': lib.halofit,
    'linear': lib.linear,
    'emu': lib.emu,
    'calculator': lib.pknl_from_input
}

baryons_power_spectrum_types = {
    'nobaryons': lib.nobaryons,
    'bcm': lib.bcm
}

# List which transfer functions can be used with the muSigma_MG
# parameterisation of modified gravity

mass_function_types = {
    'angulo': lib.angulo,
    'tinker': lib.tinker,
    'tinker10': lib.tinker10,
    'watson': lib.watson,
    'shethtormen': lib.shethtormen
}

halo_concentration_types = {
    'bhattacharya2011': lib.bhattacharya2011,
    'duffy2008': lib.duffy2008,
    'constant_concentration': lib.constant_concentration,
}

emulator_neutrinos_types = {
    'strict': lib.emu_strict,
    'equalize': lib.emu_equalize
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

    .. note:: After instantiation, you can set parameters related to the
              internal splines and numerical integration accuracy by setting
              the values of the attributes of
              :obj:`Cosmology.cosmo.spline_params` and
              :obj:`Cosmology.cosmo.gsl_params`. For example, you can set
              the generic relative accuracy for integration by executing
              ``c = Cosmology(...); c.cosmo.gsl_params.INTEGRATION_EPSREL \
= 1e-5``.
              See the module level documetaion of `pyccl.core` for details.

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
        m_nu_type (:obj:`str`, optional): The type of massive neutrinos. Should
            be one of 'inverted', 'normal', 'equal', 'single', or 'list'.
            The default of None is the same as 'normal'.
        w0 (:obj:`float`, optional): First order term of dark energy equation
            of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy equation
            of state. Defaults to 0.
        T_CMB (:obj:`float`): The CMB temperature today. The default of
            ``None`` uses the global CCL value in
            ``pyccl.physical_constants.T_CMB``.
        bcm_log10Mc (:obj:`float`, optional): One of the parameters of the
            BCM model. Defaults to `np.log10(1.2e14)`.
        bcm_etab (:obj:`float`, optional): One of the parameters of the BCM
            model. Defaults to 0.5.
        bcm_ks (:obj:`float`, optional): One of the parameters of the BCM
            model. Defaults to 55.0.
        mu_0 (:obj:`float`, optional): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        sigma_0 (:obj:`float`, optional): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        c1_mg (:obj:`float`, optional): MG parameter that enters in the scale
            dependence of mu affecting its large scale behavior. Default to 1.
            See, e.g., Eqs. (46) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        c2_mg (:obj:`float`, optional): MG parameter that enters in the scale
            dependence of Sigma affecting its large scale behavior. Default 1.
            See, e.g., Eqs. (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        lambda_mg (:obj:`float`, optional): MG parameter that sets the start
            of dependance on c1 and c2 MG parameters. Defaults to 0.0
            See, e.g., Eqs. (46) & (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        df_mg (array_like, optional): Perturbations to the GR growth rate as
            a function of redshift :math:`\\Delta f`. Used to implement simple
            modified growth scenarios.
        z_mg (array_like, optional): Array of redshifts corresponding to df_mg.
        transfer_function (:obj:`str`, optional): The transfer function to
            use. Defaults to 'boltzmann_camb'.
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
            m_nu_type = 'equal', and 'equalize', which will redistribute
            masses to be equal right before calling the emualtor but results in
            internal inconsistencies. Defaults to 'strict'.
    """
    def __init__(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None,
            Omega_k=0., Omega_g=None, Neff=3.046, m_nu=0., m_nu_type=None,
            w0=-1., wa=0., T_CMB=None,
            bcm_log10Mc=np.log10(1.2e14), bcm_etab=0.5,
            bcm_ks=55., mu_0=0., sigma_0=0.,
            c1_mg=1., c2_mg=1., lambda_mg=0., z_mg=None, df_mg=None,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='halofit',
            baryons_power_spectrum='nobaryons',
            mass_function='tinker10',
            halo_concentration='duffy2008',
            emulator_neutrinos='strict'):

        # going to save these for later
        self._params_init_kwargs = dict(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, Omega_g=Omega_g, Neff=Neff, m_nu=m_nu,
            m_nu_type=m_nu_type, w0=w0, wa=wa, T_CMB=T_CMB,
            bcm_log10Mc=bcm_log10Mc,
            bcm_etab=bcm_etab, bcm_ks=bcm_ks, mu_0=mu_0, sigma_0=sigma_0,
            c1_mg=c1_mg, c2_mg=c2_mg, lambda_mg=lambda_mg,
            z_mg=z_mg, df_mg=df_mg)

        self._config_init_kwargs = dict(
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum,
            baryons_power_spectrum=baryons_power_spectrum,
            mass_function=mass_function,
            halo_concentration=halo_concentration,
            emulator_neutrinos=emulator_neutrinos)

        self._build_cosmo()

        self._has_pk_lin = False
        self._pk_lin = {}
        self._has_pk_nl = False
        self._pk_nl = {}

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
        status = lib.parameters_write_yaml(self._params, filename, status)

        # Check status
        if status != 0:
            raise IOError("Unable to write YAML file {}".format(filename))

    @classmethod
    def read_yaml(cls, filename, **kwargs):
        """Read the parameters from a YAML file.

        Args:
            filename (:obj:`str`) Filename to read parameters from.
            **kwargs (dict) Additional keywords that supersede file contents
        """
        with open(filename, 'r') as fp:
            params = yaml.load(fp, Loader=yaml.Loader)

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
            bcm_ks=params['bcm_ks'],
            mu_0=params['mu_0'],
            sigma_0=params['sigma_0'],
            c1_mg=params['c1_mg'],
            c2_mg=params['c2_mg'],
            lambda_mg=params['lambda_mg'])
        if 'z_mg' in params:
            inits['z_mg'] = params['z_mg']
            inits['df_mg'] = params['df_mg']

        if 'm_nu' in params:
            inits['m_nu'] = params['m_nu']
            inits['m_nu_type'] = 'list'

        inits.update(kwargs)

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
            A_s=None, Omega_k=None, Neff=None, m_nu=None, m_nu_type=None,
            w0=None, wa=None, T_CMB=None,
            bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            mu_0=None, sigma_0=None, c1_mg=None, c2_mg=None, lambda_mg=None,
            z_mg=None, df_mg=None, Omega_g=None):
        """Build a ccl_parameters struct"""

        # Check to make sure Omega_k is within reasonable bounds.
        if Omega_k is not None and Omega_k < -1.0135:
            raise ValueError("Omega_k must be more than -1.0135.")

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

        # Make sure the neutrino parameters are consistent
        # and if a sum is given for mass, split into three masses.
        if hasattr(m_nu, "__len__"):
            if (len(m_nu) != 3):
                raise ValueError("m_nu must be a float or array-like object "
                                 "with length 3.")
            elif m_nu_type in ['normal', 'inverted', 'equal']:
                raise ValueError(
                    "m_nu_type '%s' cannot be passed with a list "
                    "of neutrino masses, only with a sum." % m_nu_type)
            elif m_nu_type is None:
                m_nu_type = 'list'  # False
            mnu_list = [0]*3
            for i in range(0, 3):
                mnu_list[i] = m_nu[i]

        else:
            try:
                m_nu = float(m_nu)
            except Exception:
                raise ValueError(
                    "m_nu must be a float or array-like object with "
                    "length 3.")

            if m_nu_type is None:
                m_nu_type = 'normal'
            m_nu = [m_nu]
            if (m_nu_type == 'normal'):
                if (m_nu[0] < (np.sqrt(7.62E-5) + np.sqrt(2.55E-3))
                        and (m_nu[0] > 1e-15)):
                    raise ValueError("if m_nu_type is 'normal', we are "
                                     "using the normal hierarchy and so "
                                     "m_nu must be greater than (~)0.0592 "
                                     "(or zero)")

                # Split the sum into 3 masses under normal hierarchy.
                if (m_nu[0] > 1e-15):
                    mnu_list = [0]*3
                    # This is a starting guess.
                    mnu_list[0] = 0.
                    mnu_list[1] = np.sqrt(7.62E-5)
                    mnu_list[2] = np.sqrt(2.55E-3)
                    sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]
                    # This is the Newton's method
                    while (np.abs(m_nu[0] - sum_check) > 1e-15):
                        dsdm1 = (1. + mnu_list[0] / mnu_list[1]
                                 + mnu_list[0] / mnu_list[2])
                        mnu_list[0] = mnu_list[0] - (sum_check
                                                     - m_nu[0]) / dsdm1
                        mnu_list[1] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              + 7.62E-5)
                        mnu_list[2] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              + 2.55E-3)
                        sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]

            elif (m_nu_type == 'inverted'):
                if (m_nu[0] < (np.sqrt(2.43e-3 - 7.62e-5) + np.sqrt(2.43e-3))
                        and (m_nu[0] > 1e-15)):
                    raise ValueError("if m_nu_type is 'inverted', we "
                                     "are using the inverted hierarchy "
                                     "and so m_nu must be greater than "
                                     "(~)0.0978 (or zero)")
                # Split the sum into 3 masses under inverted hierarchy.
                if (m_nu[0] > 1e-15):
                    mnu_list = [0]*3
                    mnu_list[0] = 0.  # This is a starting guess.
                    mnu_list[1] = np.sqrt(2.43e-3 - 7.62E-5)
                    mnu_list[2] = np.sqrt(2.43e-3)
                    sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]
                    # This is the Newton's method
                    while (np.abs(m_nu[0] - sum_check) > 1e-15):
                        dsdm1 = (1. + (mnu_list[0] / mnu_list[1])
                                 + (mnu_list[0] / mnu_list[2]))
                        mnu_list[0] = mnu_list[0] - (sum_check
                                                     - m_nu[0]) / dsdm1
                        mnu_list[1] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              + 7.62E-5)
                        mnu_list[2] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              - 2.43e-3)
                        sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]
            elif (m_nu_type == 'equal'):
                mnu_list = [0]*3
                mnu_list[0] = m_nu[0]/3.
                mnu_list[1] = m_nu[0]/3.
                mnu_list[2] = m_nu[0]/3.
            elif (m_nu_type == 'single'):
                mnu_list = [0]*3
                mnu_list[0] = m_nu[0]
                mnu_list[1] = 0.
                mnu_list[2] = 0.

        # Check which of the neutrino species are non-relativistic today
        N_nu_mass = 0
        if (np.abs(np.amax(m_nu) > 1e-15)):
            for i in range(0, 3):
                if (mnu_list[i] > 0.00017):  # Lesgourges et al. 2012
                    N_nu_mass = N_nu_mass + 1
            N_nu_rel = Neff - (N_nu_mass * 0.71611**4 * (4./11.)**(-4./3.))
            if N_nu_rel < 0.:
                raise ValueError("Neff and m_nu must result in a number "
                                 "of relativistic neutrino species greater "
                                 "than or equal to zero.")

        # Fill an array with the non-relativistic neutrino masses
        if N_nu_mass > 0:
            mnu_final_list = [0]*N_nu_mass
            relativistic = [0]*3
            for i in range(0, N_nu_mass):
                for j in range(0, 3):
                    if (mnu_list[j] > 0.00017 and relativistic[j] == 0):
                        relativistic[j] = 1
                        mnu_final_list[i] = mnu_list[j]
                        break
        else:
            mnu_final_list = [0.]

        # Check if any compulsory parameters are not set
        compul = [Omega_c, Omega_b, Omega_k, w0, wa, h, norm_pk,
                  n_s]
        names = ['Omega_c', 'Omega_b', 'Omega_k', 'w0', 'wa',
                 'h', 'norm_pk', 'n_s']
        for nm, item in zip(names, compul):
            if item is None:
                raise ValueError("Necessary parameter '%s' was not set "
                                 "(or set to None)." % nm)

        # Create new instance of ccl_parameters object
        # Create an internal status variable; needed to check massive neutrino
        # integral.
        T_CMB_old = lib.cvar.constants.T_CMB
        try:
            if T_CMB is not None:
                lib.cvar.constants.T_CMB = T_CMB
            status = 0
            if nz_mg == -1:
                # Create ccl_parameters without modified growth
                self._params, status = lib.parameters_create_nu(
                    Omega_c, Omega_b, Omega_k, Neff,
                    w0, wa, h, norm_pk, n_s, bcm_log10Mc,
                    bcm_etab, bcm_ks, mu_0, sigma_0, c1_mg,
                    c2_mg, lambda_mg, mnu_final_list, status)
            else:
                # Create ccl_parameters with modified growth arrays
                self._params, status = lib.parameters_create_nu_vec(
                    Omega_c, Omega_b, Omega_k, Neff, w0, wa, h,
                    norm_pk, n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                    mu_0, sigma_0, c1_mg, c2_mg, lambda_mg, z_mg,
                    df_mg, mnu_final_list, status)
            check(status)
        finally:
            lib.cvar.constants.T_CMB = T_CMB_old

        if Omega_g is not None:
            total = self._params.Omega_g + self._params.Omega_l
            self._params.Omega_g = Omega_g
            self._params.Omega_l = total - Omega_g

    def __getitem__(self, key):
        """Access parameter values by name."""
        try:
            if key == 'm_nu':
                val = lib.parameters_get_nu_masses(self._params, 3)
            else:
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
            if (self.cosmo is not None and
                    hasattr(lib, 'cosmology_free') and
                    lib.cosmology_free is not None):
                lib.cosmology_free(self.cosmo)
        if hasattr(self, "_params"):
            if (self._params is not None and
                    hasattr(lib, 'parameters_free') and
                    lib.parameters_free is not None):
                lib.parameters_free(self._params)

        # finally delete some attributes we don't want to be around for safety
        # when the context manager exits or if __del__ is called twice
        if hasattr(self, "cosmo"):
            delattr(self, "cosmo")
        if hasattr(self, "_params"):
            delattr(self, "_params")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Free the C memory this object is managing when the context manager
        exits."""
        self.__del__()

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
            if k not in ['m_nu', 'm_nu_type', 'z_mg', 'df_mg'])

        if hasattr(self._params_init_kwargs['m_nu'], '__len__'):
            string += ", m_nu=[%s, %s, %s]" % tuple(
                self._params_init_kwargs['m_nu'])
        else:
            string += ', m_nu=%s' % self._params_init_kwargs['m_nu']

        if self._params_init_kwargs['m_nu_type'] is not None:
            string += (
                ", m_nu_type='%s'" % self._params_init_kwargs['m_nu_type'])
        else:
            string += ', m_nu_type=None'

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
        if self.has_distances:
            return
        status = 0
        status = lib.cosmology_compute_distances(self.cosmo, status)
        check(status, self)

    def compute_growth(self):
        """Compute the growth function."""
        if self.has_growth:
            return
        if self['N_nu_mass'] > 0:
            warnings.warn(
                "CCL does not properly compute the linear growth rate in "
                "cosmological models with massive neutrinos!",
                category=CCLWarning)

            if self._params_init_kwargs['df_mg'] is not None:
                warnings.warn(
                    "Modified growth rates via the `df_mg` keyword argument "
                    "cannot be consistently combined with cosmological models "
                    "with massive neutrinos in CCL!",
                    category=CCLWarning)

            if (self._params_init_kwargs['mu_0'] > 0 or
                    self._params_init_kwargs['sigma_0'] > 0):
                warnings.warn(
                    "Modified growth rates via the mu-Sigma model "
                    "cannot be consistently combined with cosmological models "
                    "with massive neutrinos in CCL!",
                    category=CCLWarning)

        status = 0
        status = lib.cosmology_compute_growth(self.cosmo, status)
        check(status, self)

    def compute_linear_power(self):
        """Compute the linear power spectrum."""
        if self.has_linear_power:
            return

        if (self['N_nu_mass'] > 0 and
                self._config_init_kwargs['transfer_function'] in
                ['bbks', 'eisenstein_hu']):
            warnings.warn(
                "The '%s' linear power spectrum model does not properly "
                "account for massive neutrinos!" %
                self._config_init_kwargs['transfer_function'],
                category=CCLWarning)

        if self._config_init_kwargs['matter_power_spectrum'] == 'emu':
            warnings.warn(
                "None of the linear power spectrum models in CCL are "
                "consistent with that implictly used in the emulated "
                "non-linear power spectrum!",
                category=CCLWarning)

        # needed to init some models
        self.compute_growth()

        # Populate power spectrum splines
        trf = self._config_init_kwargs['transfer_function']
        pk = None
        rescale_s8 = True
        rescale_mg = True
        if trf is None:
            raise CCLError("You want to compute the linear power spectrum, "
                           "but you selected `transfer_function=None`.")
        elif trf == 'boltzmann_class':
            pk = get_class_pk_lin(self)
        elif trf == 'boltzmann_isitgr':
            rescale_mg = False
            pk = get_isitgr_pk_lin(self)
        elif trf == 'boltzmann_camb':
            pk = get_camb_pk_lin(self)
        elif trf in ['bbks', 'eisenstein_hu']:
            rescale_s8 = False
            rescale_mg = False
            pk = Pk2D.pk_from_model(self,
                                    model=trf)

        # Rescale by sigma8/mu-sigma if needed
        if pk:
            status = 0
            status = lib.rescale_linpower(self.cosmo, pk.psp,
                                          int(rescale_mg),
                                          int(rescale_s8),
                                          status)
            check(status, self)

        # Assign
        self._pk_lin['delta_matter:delta_matter'] = pk
        if pk:
            self._has_pk_lin = True

    def _get_halo_model_nonlin_power(self):
        from . import halos as hal
        mdef = hal.MassDef('vir', 'matter')
        conc = self._config.halo_concentration_method
        mfm = self._config.mass_function_method

        if conc == lib.bhattacharya2011:
            c = hal.ConcentrationBhattacharya13(mdef=mdef)
        elif conc == lib.duffy2008:
            c = hal.ConcentrationDuffy08(mdef=mdef)
        elif conc == lib.constant_concentration:
            c = hal.ConcentrationConstant(c=4., mdef=mdef)

        if mfm == lib.tinker10:
            hmf = hal.MassFuncTinker10(self, mass_def=mdef,
                                       mass_def_strict=False)
            hbf = hal.HaloBiasTinker10(self, mass_def=mdef,
                                       mass_def_strict=False)
        elif mfm == lib.shethtormen:
            hmf = hal.MassFuncSheth99(self, mass_def=mdef,
                                      mass_def_strict=False,
                                      use_delta_c_fit=True)
            hbf = hal.HaloBiasSheth99(self, mass_def=mdef,
                                      mass_def_strict=False)
        else:
            raise ValueError("Halo model spectra not available for your "
                             "current choice of mass function with the "
                             "deprecated implementation.")
        prf = hal.HaloProfileNFW(c)
        hmc = hal.HMCalculator(self, hmf, hbf, mdef)
        return hal.halomod_Pk2D(self, hmc, prf, normprof1=True)

    def compute_nonlin_power(self):
        """Compute the non-linear power spectrum."""
        if self.has_nonlin_power:
            return

        if self._config_init_kwargs['matter_power_spectrum'] != 'linear':
            if self._params_init_kwargs['df_mg'] is not None:
                warnings.warn(
                    "Modified growth rates via the `df_mg` keyword argument "
                    "cannot be consistently combined with '%s' for "
                    "computing the non-linear power spectrum!" %
                    self._config_init_kwargs['matter_power_spectrum'],
                    category=CCLWarning)

            if (self._params_init_kwargs['mu_0'] != 0 or
                    self._params_init_kwargs['sigma_0'] != 0):
                warnings.warn(
                    "mu-Sigma modified cosmologies "
                    "cannot be consistently combined with '%s' "
                    "for computing the non-linear power spectrum!" %
                    self._config_init_kwargs['matter_power_spectrum'],
                    category=CCLWarning)

        if (self['N_nu_mass'] > 0 and
                self._config_init_kwargs['baryons_power_spectrum'] == 'bcm'):
            warnings.warn(
                "The BCM baryonic correction model's default parameters "
                "were not calibrated for cosmological models with "
                "massive neutrinos!",
                category=CCLWarning)

        self.compute_distances()

        # Populate power spectrum splines
        mps = self._config_init_kwargs['matter_power_spectrum']
        # needed for halofit, halomodel and linear options
        if (mps != 'emu') and (mps is not None):
            self.compute_linear_power()

        pk = None
        if mps is None:
            raise CCLError("You want to compute the non-linear power "
                           "spectrum, but you selected "
                           "`matter_power_spectrum=None`.")
        elif mps == 'halo_model':
            warnings.warn(
                "The halo model option for the internal CCL matter power "
                "spectrum is deprecated. Use the more general functionality "
                "in the `halos` module.", category=CCLWarning)
            pk = self._get_halo_model_nonlin_power()
        elif mps == 'halofit':
            pkl = self._pk_lin['delta_matter:delta_matter']
            if pkl is None:
                raise CCLError("The linear power spectrum is a "
                               "necessary input for halofit")
            pk = Pk2D.apply_halofit(self, pkl)
        elif mps == 'emu':
            pk = Pk2D.pk_from_model(self, model='emu')
        elif mps == 'linear':
            pk = self._pk_lin['delta_matter:delta_matter']

        # Correct for baryons if required
        if self._config_init_kwargs['baryons_power_spectrum'] == 'bcm':
            bcm_correct_pk2d(self, pk)

        # Assign
        self._pk_nl['delta_matter:delta_matter'] = pk
        if pk:
            self._has_pk_nl = True

    def compute_sigma(self):
        """Compute the sigma(M) spline."""
        if self.has_sigma:
            return

        # we need these things before building the mass function splines
        if self['N_nu_mass'] > 0:
            # these are not consistent with anything - fun
            warnings.warn(
                "All of the halo mass function, concentration, and bias "
                "models in CCL are not properly calibrated for cosmological "
                "models with massive neutrinos!",
                category=CCLWarning)

        if self._config_init_kwargs['baryons_power_spectrum'] != 'nobaryons':
            warnings.warn(
                "All of the halo mass function, concentration, and bias "
                "models in CCL are not consistently adjusted for baryons "
                "when the power spectrum is via the BCM model!",
                category=CCLWarning)

        self.compute_linear_power()
        pk = self._pk_lin['delta_matter:delta_matter']
        if pk is None:
            raise CCLError("Linear power spectrum can't be None")
        status = 0
        status = lib.cosmology_compute_sigma(self.cosmo, pk.psp, status)
        check(status, self)

    def get_linear_power(self, name='delta_matter:delta_matter'):
        """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
        the linear power spectrum with name `name`.

        Args:
            name (:obj:`str` or `None`): name of the power spectrum to
                return. If `None`, `'delta_matter:delta_matter'` will
                be used.

        Returns:
            :class:`~pyccl.pk2d.Pk2D` object containing the linear
            power spectrum with name `name`.
        """
        if name is None:
            name = 'delta_matter:delta_matter'
        if name not in self._pk_lin:
            raise KeyError("Unknown power spectrum %s." % name)
        return self._pk_lin[name]

    def get_nonlin_power(self, name='delta_matter:delta_matter'):
        """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
        the non-linear power spectrum with name `name`.

        Args:
            name (:obj:`str` or `None`): name of the power spectrum to
                return. If `None`, `'delta_matter:delta_matter'` will
                be used.

        Returns:
            :class:`~pyccl.pk2d.Pk2D` object containing the non-linear
            power spectrum with name `name`.
        """
        if name is None:
            name = 'delta_matter:delta_matter'
        if name not in self._pk_nl:
            raise KeyError("Unknown power spectrum %s." % name)
        return self._pk_nl[name]

    @property
    def has_distances(self):
        """Checks if the distances have been precomputed."""
        return bool(self.cosmo.computed_distances)

    @property
    def has_growth(self):
        """Checks if the growth function has been precomputed."""
        return bool(self.cosmo.computed_growth)

    @property
    def has_linear_power(self):
        """Checks if the linear power spectra have been precomputed."""
        return self._has_pk_lin

    @property
    def has_nonlin_power(self):
        """Checks if the non-linear power spectra have been precomputed."""
        return self._has_pk_nl

    @property
    def has_sigma(self):
        """Checks if sigma(M) is precomputed."""
        return bool(self.cosmo.computed_sigma)

    def status(self):
        """Get error status of the ccl_cosmology object.

        .. note:: The error statuses are currently under development and
                  may not be fully descriptive.

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


class CosmologyVanillaLCDM(Cosmology):
    """A cosmology with typical flat Lambda-CDM parameters (`Omega_c=0.25`,
    `Omega_b = 0.05`, `Omega_k = 0`, `sigma8 = 0.81`, `n_s = 0.96`, `h = 0.67`,
    no massive neutrinos).

    Args:
        **kwargs (dict): a dictionary of parameters passed as arguments
            to the `Cosmology` constructor. It should not contain any of
            the LambdaCDM parameters (`"Omega_c"`, `"Omega_b"`, `"n_s"`,
            `"sigma8"`, `"A_s"`, `"h"`), since these are fixed.
    """
    def __init__(self, **kwargs):
        p = {'h': 0.67,
             'Omega_c': 0.25,
             'Omega_b': 0.05,
             'n_s': 0.96,
             'sigma8': 0.81,
             'A_s': None}
        if any(k in kwargs for k in p.keys()):
            raise ValueError("You cannot change the LCDM parameters: "
                             "%s " % list(p.keys()))
        kwargs.update(p)
        super(CosmologyVanillaLCDM, self).__init__(**kwargs)


class CosmologyCalculator(Cosmology):
    """A "calculator-mode" CCL `Cosmology` object.
    This allows users to build a cosmology from a set of arrays
    describing the background expansion, linear growth factor and
    linear and non-linear power spectra, which can then be used
    to compute more complex observables (e.g. angular power
    spectra or halo-model quantities). These are stored in
    `background`, `growth`, `pk_linear` and `pk_nonlin`.

    .. note:: Although in principle these arrays should suffice
              to compute most observable quantities some
              calculations implemented in CCL (e.g. the halo
              mass function) requires knowledge of basic
              cosmological parameters such as :math:`\\Omega_M`.
              For this reason, users must pass a minimal set
              of :math:`\\Lambda` CDM cosmological parameters.

    Args:
        Omega_c (:obj:`float`): Cold dark matter density fraction.
        Omega_b (:obj:`float`): Baryonic matter density fraction.
        h (:obj:`float`): Hubble constant divided by 100 km/s/Mpc;
            unitless.
        A_s (:obj:`float`): Power spectrum normalization. Exactly
            one of A_s and sigma_8 is required.
        sigma8 (:obj:`float`): Variance of matter density
            perturbations at an 8 Mpc/h scale. Exactly one of A_s
            and sigma_8 is required.
        n_s (:obj:`float`): Primordial scalar perturbation spectral
            index.
        Omega_k (:obj:`float`, optional): Curvature density fraction.
            Defaults to 0.
        Omega_g (:obj:`float`, optional): Density in relativistic species
            except massless neutrinos. The default of `None` corresponds
            to setting this from the CMB temperature. Note that if a
            non-`None` value is given, this may result in a physically
            inconsistent model because the CMB temperature will still
            be non-zero in the parameters.
        Neff (:obj:`float`, optional): Effective number of massless
            neutrinos present. Defaults to 3.046.
        m_nu (:obj:`float`, optional): Total mass in eV of the massive
            neutrinos present. Defaults to 0.
        m_nu_type (:obj:`str`, optional): The type of massive neutrinos.
            Should be one of 'inverted', 'normal', 'equal', 'single', or
            'list'. The default of None is the same as 'normal'.
        w0 (:obj:`float`, optional): First order term of dark energy
            equation of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy
            equation of state. Defaults to 0.
        T_CMB (:obj:`float`): The CMB temperature today. The default of
            ``None`` uses the global CCL value in
            ``pyccl.physical_constants.T_CMB``.
        background (:obj:`dict`): a dictionary describing the background
            expansion. It must contain three mandatory entries: `'a'`: an
            array of monotonically ascending scale-factor values. `'chi'`:
            an array containing the values of the comoving radial distance
            (in units of Mpc) at the scale factor values stored in `a`.
            '`h_over_h0`': an array containing the Hubble expansion rate at
            the scale factor values stored in `a`, divided by its value
            today (at `a=1`).
        growth (:obj:`dict`): a dictionary describing the linear growth of
            matter fluctuations. It must contain three mandatory entries:
            `'a'`: an array of monotonically ascending scale-factor
            values. `'growth_factor'`: an array containing the values of
            the linear growth factor :math:`D(a)` at the scale factor
            values stored in `a`. '`growth_rate`': an array containing the
            growth rate :math:`f(a)\\equiv d\\log D/d\\log a` at the scale
            factor values stored in `a`.
        pk_linear (:obj:`dict`): a dictionary containing linear power
            spectra. It must contain the following mandatory entries:
            `'a'`: an array of scale factor values. `'k'`: an array of
            comoving wavenumbers in units of inverse Mpc.
            `'delta_matter:delta_matter'`: a 2D array of shape
            `(n_a, n_k)`, where `n_a` and `n_k` are the lengths of
            `'a'` and `'k'` respectively, containing the linear matter
            power spectrum :math:`P(k,a)`. This dictionary may also
            contain other entries with keys of the form `'q1:q2'`,
            containing other cross-power spectra between quantities
            `'q1'` and `'q2'`.
        pk_nonlin (:obj:`dict`): a dictionary containing non-linear
            power spectra. It must contain the following mandatory
            entries: `'a'`: an array of scale factor values.
            `'k'`: an array of comoving wavenumbers in units of
            inverse Mpc. If `nonlinear_model` is `None`, it should also
            contain `'delta_matter:delta_matter'`: a 2D array of
            shape `(n_a, n_k)`, where `n_a` and `n_k` are the lengths
            of `'a'` and `'k'` respectively, containing the non-linear
            matter power spectrum :math:`P(k,a)`. This dictionary may
            also contain other entries with keys of the form `'q1:q2'`,
            containing other cross-power spectra between quantities
            `'q1'` and `'q2'`.
        nonlinear_model (:obj:`str`, :obj:`dict` or `None`): model to
            compute non-linear power spectra. If a string, the associated
            non-linear model will be applied to all entries in `pk_linear`
            which do not appear in `pk_nonlin`. If a dictionary, it should
            contain entries of the form `'q1:q2': model`, where `model`
            is a string designating the non-linear model to apply to the
            `'q1:q2'` power spectrum, which must also be present in
            `pk_linear`. If `model` is `None`, this non-linear power
            spectrum will not be calculated. If `nonlinear_model` is
            `None`, no additional non-linear power spectra will be
            computed. The only non-linear model supported is `'halofit'`,
            corresponding to the "HALOFIT" transformation of
            Takahashi et al. 2012 (arXiv:1208.2701).
    """
    def __init__(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None, Omega_k=0., Omega_g=None,
            Neff=3.046, m_nu=0., m_nu_type=None, w0=-1., wa=0.,
            T_CMB=None, background=None, growth=None,
            pk_linear=None, pk_nonlin=None, nonlinear_model=None):
        if pk_linear:
            transfer_function = 'calculator'
        else:
            transfer_function = None
        if pk_nonlin or nonlinear_model:
            matter_power_spectrum = 'calculator'
        else:
            matter_power_spectrum = None

        # Cosmology
        super(CosmologyCalculator, self).__init__(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h,
            n_s=n_s, sigma8=sigma8, A_s=A_s,
            Omega_k=Omega_k, Omega_g=Omega_g,
            Neff=Neff, m_nu=m_nu, m_nu_type=m_nu_type,
            w0=w0, wa=wa, T_CMB=T_CMB,
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum)

        # Parse arrays
        has_bg = background is not None
        has_dz = growth is not None
        has_pklin = pk_linear is not None
        has_pknl = pk_nonlin is not None
        has_nonlin_model = nonlinear_model is not None

        if has_bg:
            self._init_bg(background)
        if has_dz:
            self._init_growth(growth)
        if has_pklin:
            self._init_pklin(pk_linear)
        if has_pknl:
            self._init_pknl(pk_nonlin, has_nonlin_model)
        self._apply_nonlinear_model(nonlinear_model)

    def _init_bg(self, background):
        # Background
        if not isinstance(background, dict):
            raise TypeError("`background` must be a dictionary.")
        if (('a' not in background) or ('chi' not in background) or
                ('h_over_h0' not in background)):
            raise ValueError("`background` must contain keys "
                             "'a', 'chi' and 'h_over_h0'")
        a = background['a']
        chi = background['chi']
        hoh0 = background['h_over_h0']
        # Check that input arrays have the same size.
        if not (a.shape == chi.shape == hoh0.shape):
            raise ValueError("Input arrays must have the same size.")
        # Check that `a` is a monotonically increasing array.
        if not np.array_equal(a, np.sort(a)):
            raise ValueError("Input scale factor array is not "
                             "monotonically increasing.")
        # Check that the last element of a_array_back is 1:
        if np.abs(a[-1]-1.0) > 1e-5:
            raise ValueError("The last element of the input scale factor"
                             "array must be 1.0.")
        status = 0
        status = lib.cosmology_distances_from_input(self.cosmo,
                                                    a, chi, hoh0,
                                                    status)
        check(status, self)

    def _init_growth(self, growth):
        # Growth
        if not isinstance(growth, dict):
            raise TypeError("`growth` must be a dictionary.")
        if (('a' not in growth) or ('growth_factor' not in growth) or
                ('growth_rate' not in growth)):
            raise ValueError("`growth` must contain keys "
                             "'a', 'growth_factor' and 'growth_rate'")
        a = growth['a']
        dz = growth['growth_factor']
        fz = growth['growth_rate']
        # Check that input arrays have the same size.
        if not (a.shape == dz.shape
                == fz.shape):
            raise ValueError("Input arrays must have the same size.")
        # Check that a_array_grth is a monotonically increasing array.
        if not np.array_equal(a,
                              np.sort(a)):
            raise ValueError("Input scale factor array is not "
                             "monotonically increasing.")
        # Check that the last element of a is 1:
        if np.abs(a[-1]-1.0) > 1e-5:
            raise ValueError("The last element of the input scale factor"
                             "array must be 1.0.")
        status = 0
        status = lib.cosmology_growth_from_input(self.cosmo,
                                                 a, dz, fz,
                                                 status)
        check(status, self)

    def _init_pklin(self, pk_linear):
        # Linear power spectrum
        if not isinstance(pk_linear, dict):
            raise TypeError("`pk_linear` must be a dictionary")
        if (('delta_matter:delta_matter' not in pk_linear) or
                ('a' not in pk_linear) or ('k' not in pk_linear)):
            raise ValueError("`pk_linear` must contain keys 'a', 'k' "
                             "and 'delta_matter:delta_matter' "
                             "(at least)")
        na = len(pk_linear['a'])
        nk = len(pk_linear['k'])
        lk = np.log(pk_linear['k'])
        pk_names = [key for key in pk_linear if key not in ('a', 'k')]
        for n in pk_names:
            qs = n.split(':')
            if len(qs) != 2:
                raise ValueError("Power spectrum label %s could " % n +
                                 "not be parsed. Label must be of the " +
                                 "form 'q1:q2'")
            pk = pk_linear[n]
            if pk.shape != (na, nk):
                raise ValueError("Power spectrum %s has shape " % n +
                                 str(pk.shape) + " but shape " +
                                 "(%d, %d) was expected." % (na, nk))
            # Spline in log-space if the P(k) is positive-definite
            use_log = np.all(pk > 0)
            if use_log:
                pk = np.log(pk)
            # Initialize and store
            pk = Pk2D(pkfunc=None,
                      a_arr=pk_linear['a'], lk_arr=lk, pk_arr=pk,
                      is_logp=use_log, extrap_order_lok=1,
                      extrap_order_hik=2, cosmo=None)
            self._pk_lin[n] = pk
        # Set linear power spectrum as initialized
        self._has_pk_lin = True

    def _init_pknl(self, pk_nonlin, has_nonlin_model):
        # Non-linear power spectrum
        if not isinstance(pk_nonlin, dict):
            raise TypeError("`pk_nonlin` must be a dictionary")
        if (('a' not in pk_nonlin) or ('k' not in pk_nonlin)):
            raise ValueError("`pk_nonlin` must contain keys "
                             "'a' and 'k' (at least)")
        if ((not has_nonlin_model) and
                ('delta_matter:delta_matter' not in pk_nonlin)):
            raise ValueError("`pk_nonlin` must contain key "
                             "'delta_matter:delta_matter' or "
                             "use halofit to compute it")
        na = len(pk_nonlin['a'])
        nk = len(pk_nonlin['k'])
        lk = np.log(pk_nonlin['k'])
        pk_names = [key for key in pk_nonlin if key not in ('a', 'k')]
        for n in pk_names:
            qs = n.split(':')
            if len(qs) != 2:
                raise ValueError("Power spectrum label %s could " % n +
                                 "not be parsed. Label must be of the " +
                                 "form 'q1:q2'")
            pk = pk_nonlin[n]
            if pk.shape != (na, nk):
                raise ValueError("Power spectrum %s has shape " % n +
                                 str(pk.shape) + " but shape " +
                                 "(%d, %d) was expected." % (na, nk))
            # Spline in log-space if the P(k) is positive-definite
            use_log = np.all(pk > 0)
            if use_log:
                pk = np.log(pk)
            # Initialize and store
            pk = Pk2D(pkfunc=None,
                      a_arr=pk_nonlin['a'], lk_arr=lk, pk_arr=pk,
                      is_logp=use_log, extrap_order_lok=1,
                      extrap_order_hik=2, cosmo=None)
            self._pk_nl[n] = pk
        # Set non-linear power spectrum as initialized
        self._has_pk_nl = True

    def _apply_nonlinear_model(self, nonlin_model):
        if nonlin_model is None:
            return
        elif isinstance(nonlin_model, str):
            if not self._pk_lin:
                raise ValueError("You asked to use the non-linear "
                                 "model " + nonlin_model + " but "
                                 "provided no linear power spectrum "
                                 "to apply it to.")
            nld = {n: nonlin_model
                   for n in self._pk_lin}
        elif isinstance(nonlin_model, dict):
            nld = nonlin_model
        else:
            raise TypeError("`nonlinear_model` must be a string, "
                            "a dictionary or `None`")

        for name, model in nld.items():
            if name in self._pk_nl:
                continue

            if name not in self._pk_lin:
                raise KeyError(name + " is not a "
                               "known linear power spectrum")

            if ((name == 'delta_matter:delta_matter') and
                    (model is None)):
                raise ValueError("The non-linear matter power spectrum "
                                 "can't be `None`")

            if model == 'halofit':
                pkl = self._pk_lin[name]
                self._pk_nl[name] = Pk2D.apply_halofit(self, pkl)
            elif model is None:
                pass
            else:
                raise KeyError(model + " is not a valid "
                               "non-linear model.")
        # Set non-linear power spectrum as initialized
        self._has_pk_nl = True
