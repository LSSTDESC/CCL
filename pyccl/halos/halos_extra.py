from .profiles import HaloProfile, HaloProfileNFW
from .concentration import Concentration
from .profiles_2pt import Profile2pt
from ..background import h_over_h0
from scipy.special import erf
import numpy as np


class ConcentrationDuffy08M500c(Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486) extended to Delta = 500-critical.
    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08M500c'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08M500c, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(500, 'critical')

    def _check_mdef(self, mdef):
        if (mdef.Delta != 500) or (mdef.rho_type != 'critical'):
            return True
        return False

    def _setup(self):
        self.A = 3.67
        self.B = -0.0903
        self.C = -0.51

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)


class HaloProfileHOD(HaloProfileNFW):
    def __init__(self, c_M_relation,
                 lMmin=12.0, lMminp=0.,
                 lM0=12.0, lM0p=0.,
                 lM1=13.0, lM1p=0.,
                 sigmaLogM=0.4, alpha=1.,
                 a_pivot=1.):
        self.lMmin=lMmin
        self.lMminp=lMminp
        self.lM0=lM0
        self.lM0p=lM0p
        self.lM1=lM1
        self.lM1p=lM1p
        self.a0 = a_pivot
        self.sigmaLogM = sigmaLogM
        self.alpha = alpha
        super(HaloProfileHOD, self).__init__(c_M_relation)
        self._fourier = self._fourier_analytic_hod

    def update_parameters(self, **kwargs):
        self.lMmin = kwargs.get('lMmin', self.lMmin)
        self.lMminp = kwargs.get('lMminp', self.lMminp)
        self.lM0 = kwargs.get('lM0', self.lM0)
        self.lM0p = kwargs.get('lM0p', self.lM0p)
        self.lM1 = kwargs.get('lM1', self.lM1)
        self.lM1p = kwargs.get('lM1p', self.lM1p)
        self.a0 = kwargs.get('a_pivot', self.a0)
        self.sigmaLogM = kwargs.get('sigmaLogM', self.sigmaLogM)
        self.alpha = kwargs.get('alpha', self.alpha)

    def _lMmin(self, a):
        return self.lMmin + self.lMminp * (a - self.a0)

    def _lM0(self, a):
        return self.lM0 + self.lM0p * (a - self.a0)

    def _lM1(self, a):
        return self.lM1 + self.lM1p * (a - self.a0)

    def _Nc(self, M, a):
        # Number of centrals
        Mmin = 10.**self._lMmin(a)
        return 0.5 * (1 + erf(np.log10(M / Mmin) / self.sigmaLogM))

    def _Ns(self, M, a):
        # Number of satellites
        M0 = 10.**self._lM0(a)
        M1 = 10.**self._lM1(a)
        return np.heaviside(M-M0,1) * ((M - M0) / M1)**self.alpha

    def _fourier_analytic_hod(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Nc[:, None] * (1 + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Ns[:, None] * uk
        prof = Nc[:, None] * (2 * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileArnaud(HaloProfile):
    def __init__(self, b_hydro=0.2, rrange=(1e-3, 10), qpoints=100):
        self.c500 = 1.81
        self.alpha = 1.33
        self.beta = 4.13
        self.gamma = 0.31
        self.rrange = rrange
        self.qpoints = qpoints
        self.b_hydro = b_hydro

        # Interpolator for dimensionless Fourier-space profile
        self._fourier_interp = self._integ_interp()
        super(HaloProfileArnaud, self).__init__()

    def update_parameters(self, **kwargs):
        self.b_hydro = kwargs.get('b_hydro', self.b_hydro)

    def _form_factor(self, x):
        f1 = (self.c500*x)**(-self.gamma)
        f2 = (1+(self.c500*x)**self.alpha)**(-(self.beta-self.gamma)/self.alpha)
        return f1*f2

    def _integ_interp(self):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from numpy.linalg import lstsq

        def integrand(x):
            return self._form_factor(x)*x

        # # Integration Boundaries # #
        rmin, rmax = self.rrange
        lgqmin, lgqmax = np.log10(1/rmax), np.log10(1/rmin)  # log10 bounds

        q_arr = np.logspace(lgqmin, lgqmax, self.qpoints)
        f_arr = np.array([quad(integrand,
                               a=1e-4, b=np.inf,     # limits of integration
                               weight="sin",  # fourier sine weight
                               wvar=q)[0] / q
                          for q in q_arr])

        F2 = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic")

        # # Extrapolation # #
        # Backward Extrapolation
        def F1(x):
            if np.ndim(x) == 0:
                return f_arr[0]
            else:
                return f_arr[0] * np.ones_like(x)  # constant value

        # Forward Extrapolation
        # linear fitting
        Q = np.log10(q_arr[q_arr > 1e2])
        F = np.log10(f_arr[q_arr > 1e2])
        A = np.vstack([Q, np.ones(len(Q))]).T
        m, c = lstsq(A, F, rcond=None)[0]

        def F3(x):
            return 10**(m*x+c)  # logarithmic drop

        def F(x):
            return np.piecewise(x,
                                [x < lgqmin,        # backward extrapolation
                                 (lgqmin <= x)*(x <= lgqmax),  # common range
                                 lgqmax < x],       # forward extrapolation
                                [F1, F2, F3])
        return F

    def _norm(self, cosmo, M, a, b):
        """Computes the normalisation factor of the Arnaud profile.
        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
        (Arnaud et al., 2009)
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        P0 = 6.41  # reference pressure

        K = 1.65*h70*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        PM = (M*(1-b))**(2/3+aP)             # mass dependence
        Pz = h_over_h0(cosmo, a)**(8/3)  # scale factor (z) dependence

        P = K * PM * Pz
        return P

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        # hydrostatic bias
        b = self.b_hydro
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use * (1-b), a) / a

        nn = self._norm(cosmo, M_use, a, b)
        prof = self._form_factor(r_use[None, :] * R[:, None])
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        """Computes the Fourier transform of the Arnaud profile.
        .. note:: Output units are ``[norm] Mpc^3``
        """
        # Input handling
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # hydrostatic bias
        b = self.b_hydro
        # R_Delta*(1+z)
        R = mass_def.get_radius(cosmo, M_use * (1-b), a) / a

        ff = self._fourier_interp(np.log10(k_use[None, :] * R[:, None]))
        nn = self._norm(cosmo, M_use, a, b)

        prof = (4*np.pi*R**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class Profile2ptHOD(Profile2pt):
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        return prof._fourier_variance(cosmo, k, M ,a, mass_def)


class Profile2ptR(Profile2pt):
    def __init__(self, r_corr=0., r_corr_name='r_corr'):
        self.r_corr = r_corr
        self.r_corr_name = r_corr_name

    def update_parameters(self, **kwargs):
        self.r_corr = kwargs.get(self.r_corr_name, self.r_corr)

    def fourier_2pt(self, prof, cosmo, k, M, a, r_corr=0.,
                    prof2=None, mass_def=None):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)

        if prof2 is None:
            uk2 = uk1
        else:
            if not isinstance(prof2, HaloProfile):
                raise TypeError("prof2 must be of type "
                                "`HaloProfile` or `None`")

            uk2 = prof2.fourier(cosmo, k, M, a, mass_def=mass_def)

        return uk1 * uk2 * (1 + self.r_corr)
