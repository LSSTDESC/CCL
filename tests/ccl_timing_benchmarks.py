import numpy as np
import pyccl as ccl
import timeit

##### 
# This script tests the speed of CCL on key CCL functions.

def run_distance():
    """
    Run an example luminosity distance calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, 
                          sigma8=0.8) #CCL1
    a = np.linspace(0.1, 1., 200)
    dl = ccl.luminosity_distance(cosmo, a)
    return dl


def run_matter_power(mode='linear', transfer_fn='boltzmann', mpk_fn='halofit'):
    """
    Run an example matter power spectrum calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, 
                          sigma8=0.8,
                          transfer_function=transfer_fn,
                          matter_power_spectrum=mpk_fn) #CCL1
    k = np.logspace(-4., 0., 200)
    
    if mode == 'linear':
        pk = ccl.linear_matter_power(cosmo, k, a=1.)
    else:
        pk = ccl.nonlin_matter_power(cosmo, k, a=1.)
    return pk

def run_matter_power_emu(mode='nonlin', transfer_fn='emulator', mpk_fn='emu'):
    """
    Run an example matter power spectrum calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=3.2759e-01, Omega_b=5.9450e-02, h=6.1670e-01,
                          sigma8=8.7780e-01,n_s=9.6110e-01,w0=-7.0000e-01,wa=6.7220e-01, Neff=3.04,
                          transfer_function=transfer_fn,
                          matter_power_spectrum=mpk_fn)#M1
    k = np.logspace(-4., 0., 200)
    
    if mode == 'linear':
        pk = ccl.linear_matter_power(cosmo, k, a=1.)
    else:
        pk = ccl.nonlin_matter_power(cosmo, k, a=1.)
    return pk


def run_matter_power_nu_eq(mode='linear', transfer_fn='boltzmann', mpk_fn='halofit'):
    """
    Run an example matter power spectrum calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,sigma8=0.8, n_s=0.96,m_nu=np.array([0.04,0.0,0.0]),
                          transfer_function=transfer_fn,
                          matter_power_spectrum=mpk_fn) #CCL7
    k = np.logspace(-4., 0., 200)
    
    if mode == 'linear':
        pk = ccl.linear_matter_power(cosmo, k, a=1.)
    else:
        pk = ccl.nonlin_matter_power(cosmo, k, a=1.)
    return pk


def run_matter_power_nu_uneq(mode='linear', transfer_fn='boltzmann', mpk_fn='halofit'):
    """
    Run an example matter power spectrum calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,sigma8=0.8, n_s=0.96,w0=-0.9,
                         wa=0.1,m_nu=np.array([0.03,0.02,0.04]),
                        transfer_function=transfer_fn,
                          matter_power_spectrum=mpk_fn) #CCL9
    k = np.logspace(-4., 0., 200)
    
    if mode == 'linear':
        pk = ccl.linear_matter_power(cosmo, k, a=1.)
    else:
        pk = ccl.nonlin_matter_power(cosmo, k, a=1.)
    return pk

def run_matter_power_emu_nu(mode='nonlin', transfer_fn='emulator', mpk_fn='emu'):
    """
    Run an example matter power spectrum calculation.
    """
    Mnu_out = ccl.nu_masses(2.0317e-02*(5.9020e-01)**2, 'equal');
    cosmo = ccl.Cosmology(Omega_c=3.5595e-01, Omega_b=6.5074e-02, h= 5.9020e-01, n_s=9.5623e-01, 
                          sigma8=7.3252e-01, w0=-8.0194e-01, wa=3.6280e-01, Neff=3.04,m_nu=Mnu_out,
                          transfer_function=transfer_fn,
                          matter_power_spectrum=mpk_fn) #M38
    k = np.logspace(-4., 0., 200)
    
    if mode == 'linear':
        pk = ccl.linear_matter_power(cosmo, k, a=1.)
    else:
        pk = ccl.nonlin_matter_power(cosmo, k, a=1.)
    return pk


if __name__ == '__main__':

    # Luminosity distance
    print("Timing luminosity distance...")
    t = timeit.repeat("""dl = run_distance()""", 
                      setup="from __main__ import run_distance",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Linear matter power (Eisenstein + Hu)
    print("Timing matter power (Eisenstein + Hu)...")
    t = timeit.repeat("""pk = run_matter_power(mode='linear', 
                                               transfer_fn='eisenstein_hu')""", 
                      setup="from __main__ import run_matter_power",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Linear matter power (CLASS)
    print("Timing matter power (Boltzmann)...")
    t = timeit.repeat("""pk = run_matter_power(mode='linear', 
                                               transfer_fn='boltzmann')""", 
                      setup="from __main__ import run_matter_power",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Nonlinear matter power (CLASS + Halofit)
    print("Timing matter power (Boltzmann + Halofit)...")
    t = timeit.repeat("""pk = run_matter_power(mode='nonlin', 
                                               transfer_fn='boltzmann')""", 
                      setup="from __main__ import run_matter_power",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Nonlinear matter power (CosmicEmu)
    print("Timing matter power (cosmicemu)...")
    t = timeit.repeat("""pk = run_matter_power_emu(mode='nonlin', 
                                               transfer_fn='emulator',
                                               mpk_fn='emu')""", 
                      setup="from __main__ import run_matter_power_emu",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Nonlinear matter power (CosmicEmu with neutrinos)
    print("Timing matter power (cosmicemu with neutrinos)...")
    t = timeit.repeat("""pk = run_matter_power_emu_nu(mode='nonlin', 
                                               transfer_fn='emulator',
                                               mpk_fn='emu')""", 
                      setup="from __main__ import run_matter_power_emu_nu",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Nonlinear matter power (CLASS + Halofit with neutrinos - CCL7)
    print("Timing matter power (Boltzmann + Halofit with neutrinos CCL7)...")
    t = timeit.repeat("""pk = run_matter_power_nu_eq(mode='nonlin', 
                                                  transfer_fn='boltzmann')""", 
                      setup="from __main__ import run_matter_power_nu_eq",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
    # Nonlinear matter power (CLASS + Halofit with neutrinos - CCL9)
    print("Timing matter power (Boltzmann + Halofit with neutrinos CCL9)...")
    t = timeit.repeat("""pk = run_matter_power_nu_uneq(mode='nonlin', 
                                                  transfer_fn='boltzmann')""", 
                      setup="from __main__ import run_matter_power_nu_uneq",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
