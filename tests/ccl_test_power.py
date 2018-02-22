import numpy as np
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, decorators, run_module_suite
import pyccl as ccl
import sys

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.045
N_nu_rel = 0.
h = 0.7
sigma8 = 0.83
n_s = 0.96

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1])

# List of supported transfer functions
transfer_fns = ['boltzmann', 'bbks', 'boltzmann_camb', 'boltzmann_class', 
                'camb', 'class', 'eisenstein_hu', 'emulator', 
                'fitting_function', 'none']

transfer_fns = ['boltzmann_class', 
                'class', 'eisenstein_hu', #'emulator', 
                'fitting_function',]

def all_finite(vals):
    """
    Returns True if all elements are finite (i.e. not NaN or inf).
    """
    return np.alltrue( np.isfinite(vals) )


def calc_power_spectrum(Omega_v, w0, wa, transfer_fn, matter_power, linear):
    """
    Calculate linear and nonlinear power spectrum for a given set of parameters 
    and choices of transfer function and matter power spectrum.
    """
    k = np.logspace(-4., 0., 300)
    a = np.logspace(np.log10(0.51), 0., 5) # Emulator only works at z<2
    
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v    
    
    # Create new Parameters and Cosmology objects
    cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, 
                       h=h, sigma8=sigma8, n_s=n_s, Omega_k=Omega_k,
                       w0=w0, wa=wa, transfer_function=transfer_fn,
                       matter_power_spectrum=matter_power,
                       N_nu_rel=0.00641, N_nu_mass=3, m_nu=0.06)
    
    # Calculate linear and nonlinear power spectra for each scale factor, a
    for _a in a:
        if linear:
            pk_lin = ccl.linear_matter_power(cosmo, k, _a)
            assert_(all_finite(pk_lin))
        else:
            pk_nl = ccl.nonlin_matter_power(cosmo, k, _a)
            assert_(all_finite(pk_nl))


def loop_over_params(transfer_fn, matter_power, lin):
    """
    Call the power spectrum testing function for each of a set of parameters.
    """
    print(">>> (%s; %s)" % (transfer_fn, matter_power))
    
    # Loop over parameters
    for i in [0,2]: #w0_vals.size):
        calc_power_spectrum(Omega_v_vals[i], w0_vals[i], wa_vals[i], 
                            transfer_fn=transfer_fn, matter_power=matter_power,
                            linear=lin)

@decorators.slow
def test_power_spectrum_linear():
    for tfn in transfer_fns: loop_over_params(tfn, 'linear', lin=True)

@decorators.slow
def test_power_spectrum_halofit():
    for tfn in transfer_fns: loop_over_params(tfn, 'halofit', lin=True)

@decorators.slow
def test_power_spectrum_emu():
    xtransfer_fns = ['emulator',]
    for tfn in xtransfer_fns: loop_over_params(tfn, 'emu', lin=True)

@decorators.slow
def test_nonlin_power_spectrum_linear():
    for tfn in transfer_fns: loop_over_params(tfn, 'linear', lin=False)

@decorators.slow
def test_nonlin_power_spectrum_halofit():
    for tfn in transfer_fns: loop_over_params(tfn, 'halofit', lin=False)

@decorators.slow
def test_nonlin_power_spectrum_emu():
    transfer_fns = ['emulator',]
    for tfn in transfer_fns: loop_over_params(tfn, 'emu', lin=False)

if __name__ == "__main__":
    run_module_suite(argv=sys.argv)
