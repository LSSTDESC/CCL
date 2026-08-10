import warnings
import numpy as np


PARAM_BOUNDS = {
    "Omega_m": (0.25, 0.35),
    "Omega_b": (0.04, 0.055),
    "h": (0.65, 0.73),
    "n_s": (0.95, 1.0),
    "A_s": (np.exp(2.9960)/1e10, np.exp(3.091)/1e10),  # ln(1e10 As)
    "mu": (0.9, 1.1),
    "eta": (0.9, 1.1),
    "z": (0.0, 3.0),
    "bin_index": (0, 4),
}

def check_param_ranges(cosmo, mu, eta, bin_index, zs):

    def warn(name, val, lo, hi):
        if val < lo or val > hi:
            warnings.warn(
                f"{name}={val} is outside training range [{lo}, {hi}]",
                RuntimeWarning
            )

    # cosmology
    for key in ["Omega_m", "Omega_b", "h", "n_s", "A_s"]:
        warn(key, cosmo[key], *PARAM_BOUNDS[key])

    # MG params
    warn("mu", mu, *PARAM_BOUNDS["mu"])
    warn("eta", eta, *PARAM_BOUNDS["eta"])

    # bin
    if not (0 <= bin_index <= 4):
        warnings.warn("bin_index must be between 0 and 4", RuntimeWarning)

    # redshift(s)
    for z in zs:
        warn("z", z, *PARAM_BOUNDS["z"])
        
        
        
