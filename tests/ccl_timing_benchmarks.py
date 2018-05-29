#!/usr/bin/env python
"""
Run timing benchmarks on key CCL functions.
"""
import numpy as np
import pyccl as ccl
import timeit


def run_distance():
    """
    Run an example luminosity distance calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=0.26, Omega_b=0.048, h=0.67, n_s=0.96, 
                          sigma8=0.834)
    a = np.linspace(0.1, 1., 200)
    dl = ccl.luminosity_distance(cosmo, a)
    return dl


def run_matter_power(mode='linear', transfer_fn='boltzmann', mpk_fn='halofit'):
    """
    Run an example matter power spectrum calculation.
    """
    cosmo = ccl.Cosmology(Omega_c=0.26, Omega_b=0.048, h=0.67, n_s=0.96, 
                          sigma8=0.834, Neff=3.04,
                          transfer_function=transfer_fn,
                          matter_power_spectrum=mpk_fn)
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
    t = timeit.repeat("""pk = run_matter_power(mode='nonlin', 
                                               transfer_fn='emulator',
                                               mpk_fn='emu')""", 
                      setup="from __main__ import run_matter_power",
                      number=1, repeat=20)
    print("\tRuntime: %3.3f +/- %3.3f sec" % (np.median(t), np.std(t)))
    
