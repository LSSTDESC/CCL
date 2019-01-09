import numpy as np
from numpy.testing import dec as decorators
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, run_module_suite
import pyccl as ccl
from pyccl import CCLError
import sys

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.045
Neff = 3.046
mnu_sum = 0.06
mnu_list = [0.02, 0.02, 0.02] # For use with P(k) from emulator
h = 0.7
sigma8 = 0.83
n_s = 0.96

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1])

# Non-zero values of mu_0 / sigma_0 for testing functionality of 
# mu / Sigma parameterisation of modified gravity 
# (For other cases these take default value of 0
mu_0 = 0.1
sigma_0 = -0.1


def all_finite(vals):
    """
    Returns True if all elements are finite (i.e. not NaN or inf).
    """
    return np.all( np.isfinite(vals) )


def calc_power_spectrum(Omega_v, w0, wa, transfer_fn, matter_power, linear, raise_errors):
    """
    Calculate linear and nonlinear power spectrum for a given set of parameters
    and choices of transfer function and matter power spectrum.
    """
    k = np.logspace(-5., 1., 300)
    a = np.logspace(np.log10(0.51), 0., 5) # Emulator only works at z<2

    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    if (raise_errors == False):
        if (transfer_fn == 'eisenstein_hu' or transfer_fn == 'bbks'):
            mnu = 0. # The bbks and E-H P(k) are not defined for massive neutrinos.
        elif (transfer_fn == 'emulator' and matter_power=='emu'):
            mnu = mnu_list # For the emulator, we must have 3 equal masses
        else:
            mnu = mnu_sum
    elif (raise_errors==True):
        if (transfer_fn =='eisenstein_hu' or transfer_fn =='bbks'):
            mnu = mnu_sum #Use massive neutrinos to deliberately raise an error
        elif (transfer_fn == 'emulator' and matter_power == 'emu'):
            mnu = mnu_sum #Use a sum instead of an equal list to deliberately raise an error.
        else:
            raise(ValueError, "Transfer function %s with matter power spectrum method %s has no case for which to test errors are raised." % (transfer_fn, matter_power))

    # Create a new Cosmology object
    cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b,
                       h=h, sigma8=sigma8, n_s=n_s, Omega_k=Omega_k,
                       w0=w0, wa=wa, transfer_function=transfer_fn,
                       matter_power_spectrum=matter_power,
                       Neff = Neff, m_nu = mnu)

    # Calculate linear and nonlinear power spectra for each scale factor, a
    for _a in a:
        if linear:
            if raise_errors==False:
                pk_lin = ccl.linear_matter_power(cosmo, k, _a)
                assert_(all_finite(pk_lin))
            else:
                assert_raises(CCLError, ccl.linear_matter_power, cosmo, k, _a)
        else:
            if raise_errors==False:
                pk_nl = ccl.nonlin_matter_power(cosmo, k, _a)
                assert_(all_finite(pk_nl))
            else:
                assert_raises(RuntimeError,ccl.nonlin_matter_power, cosmo, k, _a)
                
def calc_power_spectrum_muSig(transfer_fn, matter_power, linear):
    """ Check the behaviour of the calculation of the linear and 
    nonlinear power spectrum in the mu / Sigma parameterisation of
    modified gravity. """
	
    k = np.logspace(-5., 1., 300)
    a = np.logspace(np.log10(0.51), 0., 5) # Emulator only works at z<2
	
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v_vals[0]
          
    for _a in a:
        if ((transfer_fn !=	'boltzmann') and (transfer_fn != 'boltzmann_class') and (transfer_fn!='class')):
		    assert_raises(ValueError, ccl.Cosmology, Omega_c=Omega_c, Omega_b=Omega_b, 
                       h=h, sigma8=sigma8, n_s=n_s, Omega_k=Omega_k,
                       w0=w0_vals[0], wa=wa_vals[0], transfer_function=transfer_fn,
                       matter_power_spectrum=matter_power,
                       Neff = Neff, mu_0 = mu_0, sigma_0 = sigma_0)
        else:
            cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, 
                       h=h, sigma8=sigma8, n_s=n_s, Omega_k=Omega_k,
                       w0=w0_vals[0], wa=wa_vals[0], transfer_function=transfer_fn,
                       matter_power_spectrum=matter_power,
                       Neff = Neff, mu_0 = mu_0, sigma_0 = sigma_0)
			
            if linear:
                pk_lin = ccl.linear_matter_power(cosmo, k, _a)
                assert_(all_finite(pk_lin))
            else:
                if (matter_power=='linear'):
                    pk_lin = ccl.nonlin_matter_power(cosmo, k, _a)
                    assert_(all_finite(pk_lin))
                else:
					assert_raises(RuntimeError, ccl.nonlin_matter_power, cosmo, k, _a)

def loop_over_params(transfer_fn, matter_power, lin, raise_errs):
    """
    Call the power spectrum testing function for each of a set of parameters.
    """
    print(">>> (%s; %s)" % (transfer_fn, matter_power))

    # Loop over parameters
    for i in [0,2]: #w0_vals.size):
        calc_power_spectrum(Omega_v_vals[i], w0_vals[i], wa_vals[i],
                            transfer_fn=transfer_fn, matter_power=matter_power,
                            linear=lin, raise_errors = raise_errs)

def test_power_spectrum_linear():
    for tfn in ['bbks', 'eisenstein_hu']:
        loop_over_params(tfn, 'linear', lin=True, raise_errs = False)

"""@decorators.slow
def test_power_spectrum_linear_slow():
    for tfn in ['boltzmann',]:
        loop_over_params(tfn, 'linear', lin=True, raise_errs = False)"""

def test_power_spectrum_halofit():
    for tfn in ['eisenstein_hu', 'bbks',]:
        loop_over_params(tfn, 'halofit', lin=True, raise_errs = False)

"""@decorators.slow
def test_power_spectrum_halofit_slow():
    for tfn in ['boltzmann',]:
        loop_over_params(tfn, 'halofit', lin=True, raise_errs = False)

@decorators.slow
def test_power_spectrum_emu():
    for tfn in ['emulator',]: loop_over_params(tfn, 'emu', lin=True, raise_errs = False)"""

def test_nonlin_power_spectrum_linear():
    for tfn in ['eisenstein_hu', 'bbks']:
        loop_over_params(tfn, 'linear', lin=False, raise_errs = False)

"""@decorators.slow
def test_nonlin_power_spectrum_linear_slow():
    for tfn in ['boltzmann',]:
        loop_over_params(tfn, 'linear', lin=False, raise_errs = False)"""

def test_nonlin_power_spectrum_halofit():
    for tfn in ['eisenstein_hu', 'bbks']:
        loop_over_params(tfn, 'halofit', lin=False, raise_errs = False)

"""@decorators.slow
def test_nonlin_power_spectrum_halofit_slow():
    for tfn in ['boltzmann',]:
        loop_over_params(tfn, 'halofit', lin=False, raise_errs = False)

@decorators.slow
def test_nonlin_power_spectrum_emu():
    transfer_fns = ['emulator',]
    for tfn in transfer_fns: loop_over_params(tfn, 'emu', lin=False, raise_errs = False)

@decorators.slow
def test_raise_error_EH_bbks_lin():
    for tfn in ['eisenstein_hu']:
        loop_over_params(tfn, 'linear', lin=True, raise_errs=True)

@decorators.slow
def test_raise_error_EH_bbks_halofit():
    for tfn in ['bbks']:
        loop_over_params(tfn, 'halofit', lin=False, raise_errs=True)

@decorators.slow
def test_raise_error_EH_bbks_nonlin_linear():
    for tfn in ['bbks']:
        loop_over_params(tfn, 'linear', lin=False, raise_errs=True)

@decorators.slow
def test_raise_error_emu():
    transfer_fns = ['emulator',]
    for tfn in transfer_fns: loop_over_params(tfn, 'emu', lin=True, raise_errs = True)

@decorators.slow
def test_raise_error_emu_nonlin():
    transfer_fns = ['emulator',]
    for tfn in transfer_fns: loop_over_params(tfn, 'emu', lin=False, raise_errs = True)
    
@decorators.slow
def test_muSig():
    for tfn in ['eisenstein_hu', 'bbks']:
        calc_power_spectrum_muSig(tfn, 'linear', True)
        calc_power_spectrum_muSig(tfn, 'linear', False)
        
    for tfn in ['emulator']:
		calc_power_spectrum_muSig(tfn, 'emu', False)
    
    for tfn in ['boltzmann_class']:
		calc_power_spectrum_muSig(tfn, 'halofit', True)
		calc_power_spectrum_muSig(tfn, 'halofit', False)
		calc_power_spectrum_muSig(tfn, 'linear', True)
		calc_power_spectrum_muSig(tfn, 'linear', False)"""

if __name__ == "__main__":
    run_module_suite(argv=sys.argv)
