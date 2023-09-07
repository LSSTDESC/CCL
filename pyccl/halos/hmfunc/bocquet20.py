__all__ = ("MassFuncBocquet20",)

import numpy as np

from . import MassFunc


class MassFuncBocquet20(MassFunc):
    """Implements the mass function emulator of `Bocquet et al. 2020
    <https://arxiv.org/abs/2003.12116>`_. This parametrization is
    only valid for '200c' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = 'Bocquet20'

    def __init__(self, *,
                 mass_def="200c",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        import MiraTitanHMFemulator
        self.emu = MiraTitanHMFemulator.Emulator()

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != '200c'

    def __call__(self, cosmo, M, a):
        # Set up cosmology
        h = cosmo['h']
        om = cosmo['Omega_m']*h**2
        ob = cosmo['Omega_b']*h**2
        sigma8 = cosmo['sigma8']
        if np.isnan(sigma8):
            raise ValueError("sigma8 must be provided to use the MiraTitan "
                             "mass function emulator.")
        onu = cosmo.omega_x(1.0, 'neutrinos_massive')*h**2
        hmcosmo = {'Ommh2': om,
                   'Ombh2': ob,
                   'Omnuh2': onu,
                   'n_s': cosmo['n_s'],
                   'h': h,
                   'sigma_8': sigma8,
                   'w_0': cosmo['w0'],
                   'w_a': cosmo['wa']}

        # Filter out masses beyond emulator range
        M_use = np.atleast_1d(M)
        # Add h-inverse
        Mh = M_use * h
        m_hi = Mh > 1E16
        m_lo = Mh < 1E13
        m_good = ~(m_hi | m_lo)

        mfh = np.zeros_like(Mh)
        # Populate low-halo masses through extrapolation if needed
        if np.any(m_lo):
            # Evaluate slope at low masses
            m0 = 10**np.array([13.0, 13.1])
            mfp = self.emu.predict(hmcosmo, 1/a-1, m0,
                                   get_errors=False)[0].flatten()
            slope = np.log(mfp[1]/mfp[0])/np.log(m0[1]/m0[0])
            mfh[m_lo] = mfp[0]*(Mh[m_lo]/m0[0])**slope
        # Predict in good range of masses
        if np.any(m_good):
            mfp = self.emu.predict(hmcosmo, 1/a-1, Mh[m_good],
                                   get_errors=False)[0].flatten()
            mfh[m_good] = mfp
        # For masses above emulator range, n(M) will be set to zero
        # Remove h-inverse and correct for dn/dln(M) -> dn/dlog10(M)
        # log10 = 2.30258509299
        mf = mfh*h**3*2.30258509299

        if np.ndim(M) == 0:
            return mf[0]
        return mf
