__all__ = ("MassFuncNishimichi19",)

import numpy as np

from . import MassFunc


class MassFuncNishimichi19(MassFunc):
    """Implements the mass function emulator of `Nishimichi et al. 2019
    <https://arxiv.org/abs/1811.09504>`_.
    documentation is
    `here <https://dark-emulator.readthedocs.io/en/latest/index.html#>`_.
    This parametrization is only valid for '200m' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
        extrapolate (:obj:`bool`): if ``True``, extrapolation will be used
            to calculate the the mass function for small halo masses, below
            the range supported by the Dark Emulator. Otherwise, an error
            is thrown when evaluating the mass function on these masses.
    """
    name = 'Nishimichi19'

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True,
                 extrapolate=False):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        from dark_emulator import model_hod
        self.hod = model_hod.darkemu_x_hod({"fft_num": 1})
        self.extrapolate = extrapolate

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != '200m'

    def __call__(self, cosmo, M, a):
        # Set up cosmology
        h = cosmo['h']
        ob = cosmo['Omega_b']*h**2
        oc = cosmo['Omega_c']*h**2
        As = cosmo['A_s']
        if np.isnan(As):
            raise ValueError("A_s must be provided to use the Dark Emulator "
                             "halo mass function.")
        ns = cosmo['n_s']
        w0 = cosmo['w0']
        onu = 0.00064  # we fix this value (Nishimichi et al. 2019)
        Ode = 1.-(ob+oc+onu)/h**2.
        # (omega_b,omega_c,Omega_de,ln(10^10As),ns,w)
        cparam = np.array([ob, oc, Ode, np.log(As*1E10), ns, w0])
        self.hod.set_cosmology(cparam)
        # calculating interpolated dndM array is (ln(M), ln(dndM))
        self.hod._compute_dndM_spl(redshift=1./a-1.)

        # Filter out masses beyond emulator range
        M_use = np.atleast_1d(M)
        # Add h-inverse
        Mh = M_use * h
        m_hi = Mh > 1E16
        m_lo = Mh < 1E12
        m_good = ~(m_hi | m_lo)

        mfh = np.zeros_like(Mh)
        # Populate low-halo masses through extrapolation if needed
        if np.any(m_lo):
            if self.extrapolate:
                # Evaluate slope at low masses
                m0 = 10**np.array([12.0, 12.1])
                mfp = self.hod.dndM_spl(np.log(m0))
                mfp = np.exp(mfp)
                slope = np.log(mfp[1]/mfp[0])/np.log(m0[1]/m0[0])
                mfh[m_lo] = mfp[0]*(Mh[m_lo]/m0[0])**slope
            else:
                raise RuntimeError(
                    "Input mass range is not supported. "
                    "The supported range is from 10^12 to 10^16 Msun/h. "
                    "If you want to obtain mass function at "
                    "M200m<10^12 Msun/h, use 'extrapolate=True'. "
                    "This function will then extrapolate outside the "
                    "supported range.")
        # Predict in good range of masses
        if np.any(m_good):
            mfp = self.hod.dndM_spl(np.log(Mh[m_good]))
            mfp = np.exp(mfp)
            mfh[m_good] = mfp
        # For masses above emulator range, n(M) will be set to zero
        # Remove h-inverse and correct for dn/dM -> dn/dlog10(M)
        # ln10 = 2.30258509299
        mf = mfh*Mh*2.30258509299*h**3.

        if np.ndim(M) == 0:
            return mf[0]
        return mf
