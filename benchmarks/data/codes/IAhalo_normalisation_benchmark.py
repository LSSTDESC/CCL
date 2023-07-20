# Taken from github.com/c-d-leonard/IA_GGL/blob/CCL_IAhalo_bm/
# Written by Danielle Leonard.
# This is script is intended to provide code to benchmark the
# integration over the halo mass function and HOD quantities for
# the IA halo model in CCL

import numpy as np
import scipy.integrate
import pyccl as ccl


def normalisation(Mhalo, cosmo, conc, z):
    """ Returns the 'normalisation factor' that goes into the IA halo model
    power spectrum: satelite fraction * Nsat / tot_ns as a function of Mhalo.
    Mhalo is the vector of halo masses over which to integrate in units of Msol
    cosmo is a pyccl cosmology object
    conc is a concentration relation object
    z is the redshift at which to calculate this
    """

    # Get the halo mass function
    HMF_setup = ccl.halos.MassFuncTinker10(cosmo)
    HMF = HMF_setup.get_mass_function(cosmo, Mhalo, 1. / (1. + z))

    # Get HOD quantities we need
    HODHProf = ccl.halos.HaloProfileHOD(conc)
    Ncen_lens = HODHProf._Nc(Mhalo, 1. / (1. + z))
    Nsat_lens = HODHProf._Ns(Mhalo, 1. / (1. + z))

    # Get total number of satellite galaxies
    tot_ns = scipy.integrate.simps(Ncen_lens * Nsat_lens * HMF, np.log10(Mhalo))
    tot_nc = scipy.integrate.simps((Ncen_lens) * HMF, np.log10(Mhalo))

    tot_all = tot_ns + tot_nc

    # Fraction of satellites:
    f_s = tot_ns / tot_all

    return f_s / tot_ns * Nsat_lens


if (__name__ == "__main__"):
    Mhalo = np.logspace(10., 16, 30)  # Units Msol

    z = 0.0

    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_k=0, sigma8=0.81, n_s=0.96, h=1.)

    concentration = ccl.halos.ConcentrationDuffy08()

    norm_term = normalisation(Mhalo, cosmo, concentration, z)

    save = np.column_stack((Mhalo, norm_term))
    np.savetxt('../IA_halomodel_norm_term.dat', save)


