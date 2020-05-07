from .background import comoving_radial_distance, scale_factor_of_chi
from .tracers import Tracer
import numpy as np


class SZTracer(Tracer):
    def __init__(self, cosmo, z_max=6., n_chi=1024):
        self.chi_max = comoving_radial_distance(cosmo, 1./(1+z_max))
        chi_arr = np.linspace(0, self.chi_max, n_chi)
        a_arr = scale_factor_of_chi(cosmo, chi_arr) 
        # avoid recomputing every time
        # Units of eV * Mpc / cm^3

        # sigma_T = 6.65e-29 m2
        # m_e = 9.11e-31 kg
        # c = 3e8  m/s

        # eV2J = 1.6e-19 eV/J (J=kg m2/s2)
        # cm2pc = 3.1e18 cm/pc

        # prefac = (sigma_t*(10**2)**2/(m_e*c**2/J2eV))*cm2pc*10**6

        prefac = 4.01710079e-06
        w_arr = prefac * a_arr

        self._trc = []
        self.add_tracer(cosmo, kernel=(chi_arr, w_arr))
