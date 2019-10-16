from . import ccllib as lib
from .core import check
from .background import species_types, rho_x, omega_x
import numpy as np


def sigmaM(cosmo, M, a):
    """Root mean squared variance for the given halo mass of the linear power
    spectrum; Msun.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        M (float or array_like): Halo masses; Msun.
        a (float): scale factor.

    Returns:
        float or array_like: RMS variance of halo mass.
    """
    cosmo.compute_sigma()

    # sigma(M)
    logM = np.log10(np.atleast_1d(M))
    status = 0
    sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                len(logM), status)
    check(status)
    if np.isscalar(M):
        sigM = sigM[0]
    return sigM


class MassFunc(object):
    def __init__(self, name, cosmo, mass_def):
        cosmo.compute_sigma()
        self.name = name
        if self._check_mdef(mass_def):
            raise ValueError("Mass function " + name +
                             " is not compatible with mass definition" +
                             " Delta = %s, " % (mass_def.Delta) +
                             " rho = " + mass_def.rho_type)
        self.mdef = mass_def
        self._setup(cosmo)

    def _setup(self, cosmo):
        pass

    def _check_mdef(self, mdef):
        return False

    def _get_consistent_mass(self, cosmo, M, a, mdef_other):
        if mdef_other is not None:
            M_use = mdef_other.translate_mass(cosmo, M, a, self.mdef)
        else:
            M_use = M
        return np.log10(M_use)

    def get_mass_function(self, cosmo, M, a, mdef_other=None):
        M_use = np.atleast_1d(M)
        logM = self._get_consistent_mass(cosmo, M_use,
                                         a, mdef_other)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                    len(logM), status)
        check(status);
        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, logM,
                                                   len(logM), status)
        check(status)

        rho = lib.cvar.constants.RHO_CRITICAL * cosmo['Omega_m'] * cosmo['h']**2
        f = self.get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use
        
        if np.isscalar(M):
            mf = mf[0]
        return mf

    def get_fsigma(self, cosmo, sigM, a, lnM):
        raise NotImplementedError("Use one of the non-default MassFunction classes")


class MassFuncPress74(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncPress74, self).__init__("Press74",
                                              cosmo,
                                              mass_def)

    def _setup(self, cosmo):
        self.norm = np.sqrt(2/np.pi)

    def _check_mdef(self, mdef):
        if mdef.Delta!='fof':
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status);

        nu = delta_c/sigM
        return self.norm * nu * np.exp(-0.5 * nu**2)


class MassFuncSheth99(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncSheth99, self).__init__("Sheth99",
                                              cosmo,
                                              mass_def)

    def _setup(self, cosmo):
        self.A = 0.21615998645
        self.p = 0.3
        self.a = 0.707

    def _check_mdef(self, mdef):
        if mdef.Delta!='fof':
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status);

        nu = delta_c/sigM
        return nu*self.A*(1.+(self.a*nu**2)**(-self.p))*np.exp(-self.a*nu**2/2.);


class MassFuncJenkins01(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncJenkins01, self).__init__("Jenkins01",
                                                cosmo,
                                                mass_def)

    def _setup(self, cosmo):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _check_mdef(self, mdef):
        if mdef.Delta!='fof':
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.fabs(-np.log(sigM) + self.b)**self.q)


class MassFuncTinker08(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncTinker08, self).__init__("Tinker08",
                                               cosmo,
                                               mass_def)

    def _setup(self, cosmo):
        from scipy.interpolate import interp1d

        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])
        
        alpha = np.array([0.186, 0.200, 0.212, 0.218, 0.248,
                          0.255, 0.260, 0.260, 0.260])
        beta = np.array([1.47, 1.52, 1.56, 1.61, 1.87,
                         2.13, 2.30, 2.53, 2.66])
        gamma = np.array([2.57, 2.25, 2.05, 1.87, 1.59,
                          1.51, 1.46, 1.44, 1.41])
        phi = np.array([1.19, 1.27, 1.34, 1.45, 1.58,
                        1.80, 1.97, 2.24, 2.44])
        ldelta = np.log10(delta)
        ld = np.log10(self.mdef.Delta)
        self.pA0 = interp1d(ldelta, alpha)(ld)
        self.pa0 = interp1d(ldelta, beta)(ld)
        self.pd = 10.**(-(0.75/np.log10(self.mdef.Delta/75.))**1.2)
        self.pb0 = interp1d(ldelta, gamma)(ld)
        self.pc = interp1d(ldelta, phi)(ld)

    def _check_mdef(self, mdef):
        if (mdef.Delta<200.) or (mdef.Delta>3200.) or (mdef.rho_type!='matter'):
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        pA = self.pA0 * a**0.14
        pa = self.pa0 * a**0.06
        pb = self.pb0 * a**self.pd
        return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc/sigM**2)


class MassFuncDespali16(MassFunc):
    def __init__(self, cosmo, mass_def, ellipsoidal=False):
        super(MassFuncDespali16, self).__init__("Despali16",
                                                cosmo,
                                                mass_def)
        self.ellipsoidal = ellipsoidal

    def _setup(self, cosmo):
        pass

    def _check_mdef(self, mdef):
        if mdef.Delta == 'fof':
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status);

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status)

        x = np.log10(self.mdef.get_Delta(cosmo, a) *
                     omega_x(cosmo, a, self.mdef.rho_type) / Dv)

        if self.ellipsoidal:
            A = -0.1768 * x + 0.3953
            a = 0.3268 * x**2 + 0.2125 * x + 0.7057
            p = -0.04570 * x**2 + 0.1937 * x + 0.2206
        else:
            A = -0.1362 * x + 0.3292
            a = 0.4332 * x**2 + 0.2263 * x + 0.7665
            p = -0.1151 * x**2 + 0.2554 * x + 0.2488

        nu = delta_c/sigM
        nu_p = a * nu**2

        return 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)


class MassFuncTinker10(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncTinker10, self).__init__("Tinker10",
                                               cosmo,
                                               mass_def)
 
    def _setup(self, cosmo):
        from scipy.interpolate import interp1d

        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])

        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])
        alpha = np.array([0.368, 0.363, 0.385, 0.389, 0.393,
                          0.365, 0.379, 0.355, 0.327])
        beta = np.array([0.589, 0.585, 0.544, 0.543, 0.564,
                         0.623, 0.637, 0.673, 0.702])
        gamma = np.array([0.864, 0.922, 0.987, 1.09, 1.20,
                          1.34, 1.50, 1.68, 1.81])
        phi = np.array([-0.729, -0.789, -0.910, -1.05, -1.20,
                        -1.26, -1.45, -1.50, -1.49])
        eta = np.array([-0.243, -0.261, -0.261, -0.273, -0.278,
                        -0.301, -0.301, -0.319, -0.336])

        ldelta = np.log10(delta)
        ld = np.log10(self.mdef.Delta)
        self.pA0 = interp1d(ldelta, alpha)(ld)
        self.pa0 = interp1d(ldelta, eta)(ld)
        self.pb0 = interp1d(ldelta, beta)(ld)
        self.pc0 = interp1d(ldelta, gamma)(ld)
        self.pd0 = interp1d(ldelta, phi)(ld)

    def _check_mdef(self, mdef):
        if (mdef.Delta<200.) or (mdef.Delta>3200.) or (mdef.rho_type!='matter'):
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        nu = 1.686 / sigM
        pa = self.pa0 * a**(-0.27)
        pb = self.pb0 * a**(-0.20)
        pc = self.pc0 * a**0.01
        pd = self.pd0 * a**0.08
        return nu * self.pA0 * (1 + (pb * nu)**(-2 * pd)) * \
            nu**(2 * pa) * np.exp(-0.5 * pc * nu**2)


class MassFuncBocquet16(MassFunc):
    def __init__(self, cosmo, mass_def, hydro=True):
        self.hydro = hydro
        super(MassFuncBocquet16, self).__init__("Bocquet16",
                                                cosmo,
                                                mass_def)

    def _setup(self, cosmo):
        if self.mdef_type=='200m':
            if self.hydro:
                self.A0 = 0.228
                self.a0 = 2.15
                self.b0 = 1.69
                self.c0 = 1.30
                self.Az = 0.285
                self.az = -0.058
                self.bz = -0.366
                self.cz = -0.045
            else:
                self.A0 = 0.175
                self.a0 = 1.53
                self.b0 = 2.55
                self.c0 = 1.19
                self.Az = -0.012
                self.az = -0.040
                self.bz = -0.194
                self.cz = -0.021
        elif self.mdef_type=='200c':
            if self.hydro:
                self.A0 = 0.202
                self.a0 = 2.21
                self.b0 = 2.00
                self.c0 = 1.57
                self.Az = 1.147
                self.az = 0.375
                self.bz = -1.074
                self.cz = -0.196
            else:
                self.A0 = 0.222
                self.a0 = 1.71
                self.b0 = 2.24
                self.c0 = 1.46
                self.Az = 0.269
                self.az = 0.321
                self.bz = -0.621
                self.cz = -0.153
        elif self.mdef_type=='500c':
            if self.hydro:
                self.A0 = 0.180
                self.a0 = 2.29
                self.b0 = 2.44
                self.c0 = 1.97
                self.Az = 1.088
                self.az = 0.150
                self.bz = -1.008
                self.cz = -0.322
            else:
                self.A0 = 0.241
                self.a0 = 2.18
                self.b0 = 2.35
                self.c0 = 2.02
                self.Az = 0.370
                self.az = 0.251
                self.bz = -0.698
                self.cz = -0.310

    def _check_mdef(self, mdef):
        if np.fabs(mdef.Delta-200.)<1E-4:
            if mdef.rho_type=='matter':
                self.mdef_type='200m'
            elif mdef.rho_type=='critical':
                self.mdef_type='200c'
            else:
                return True
        elif np.fabs(mdef.Delta-500.)<1E-4:
            if mdef.rho_type=='critical':
                self.mdef_type='500c'
            else:
                return True
        else:
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        zp1 = 1./a
        AA = self.A0 * zp1**self.Az
        aa = self.a0 * zp1**self.az
        bb = self.b0 * zp1**self.bz
        cc = self.c0 * zp1**self.cz

        f = AA * ((sigM / bb)**-aa + 1.0) * np.exp(-cc / sigM**2)

        if self.mdef_type == '200c':
            z = 1./a-1
            Omega_m = omega_x(cosmo, a, "matter")
            gamma0 = 3.54E-2 + Omega_m**0.09
            gamma1 = 4.56E-2 + 2.68E-2 / Omega_m
            gamma2 = 0.721 + 3.50E-2 / Omega_m
            gamma3 = 0.628 + 0.164 / Omega_m
            delta0 = -1.67E-2 + 2.18E-2 * Omega_m
            delta1 = 6.52E-3 - 6.86E-3 * Omega_m
            gamma = gamma0 + gamma1 * np.exp(-((gamma2 - z) / gamma3)**2)
            delta = delta0 + delta1 * z
            M200c_M200m = gamma + delta * lnM
            f *= M200c_M200m
        elif self.mdef_type == '500c':
            z = 1./a-1
            Omega_m = omega_x(cosmo, a, "matter")
            alpha0 = 0.880 + 0.329 * Omega_m
            alpha1 = 1.00 + 4.31E-2 / Omega_m
            alpha2 = -0.365 + 0.254 / Omega_m
            alpha = alpha0 * (alpha1 * z + alpha2) / (z + alpha2)
            beta = -1.7E-2 + 3.74E-3 * Omega_m
            M500c_M200m = alpha + beta * lnM
            f *= M500c_M200m
        return f


class MassFuncWatson(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncWatson, self).__init__("Watson",
                                             cosmo,
                                             mass_def)
 
    def _setup(self, cosmo):
        self.A = [0.990, 3.216, 0.074]
        self.a = [5.907, 3.599, 2.344]
        self.b = [3.136, 3.058, 2.349]
        self.c = 1.318

    def _check_mdef(self, mdef):
        if (np.fabs(mdef.Delta - 200.) > 1E-4) or (mdef.rho_type!='matter'):
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        om = omega_x(cosmo, a, "matter")

        pA = om * (self.A[0] * a**self.A[1] + self.A[2])
        pa = om * (self.a[0] * a**self.a[1] + self.a[2])
        pb = om * (self.b[0] * a**self.b[1] + self.b[2])

        return pA * ((pb / sigM)**pa + 1.) * np.exp(-self.c / sigM**2)


class MassFuncAngulo(MassFunc):
    def __init__(self, cosmo, mass_def):
        super(MassFuncAngulo, self).__init__("Angulo",
                                             cosmo,
                                             mass_def)
 
    def _setup(self, cosmo):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _check_mdef(self, mdef):
        if (np.fabs(mdef.Delta - 200.) > 1E-4) or (mdef.rho_type!='matter'):
            return True
        return False

    def get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * (self.a / sigM + 1.)**self.b * np.exp(-self.c / sigM**2)
