class ConcentrationPrada12(Concentration):
    """ Concentration-mass relation by Prada et al. 2012
    (arXiv:1104.5130). This parametrization is only valid for
    S.O. masses with Delta = 200-critical.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Prada12'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def=None):
        super(ConcentrationPrada12, self).__init__(mass_def=mass_def)

    def _default_mass_def(self):
        self.mass_def = MassDef(200, 'critical')

    def _check_mass_def(self, mass_def):
        if isinstance(mass_def.Delta, str):
            return True
        elif not ((int(mass_def.Delta) == 200) and
                  (mass_def.rho_type == 'critical')):
            return True
        return False

    def _setup(self):
        self.c0 = 3.681
        self.c1 = 5.033
        self.al = 6.948
        self.x0 = 0.424
        self.i0 = 1.047
        self.i1 = 1.646
        self.be = 7.386
        self.x1 = 0.526
        self.cnorm = 1. / self._cmin(1.393)
        self.inorm = 1. / self._imin(1.393)

    def _cmin(self, x):
        return self.c0 + (self.c1 - self.c0) * \
            (np.arctan(self.al * (x - self.x0)) / np.pi + 0.5)

    def _imin(self, x):
        return self.i0 + (self.i1 - self.i0) * \
            (np.arctan(self.be * (x - self.x1)) / np.pi + 0.5)

    def _concentration(self, cosmo, M, a):
        sig = sigmaM(cosmo, M, a)
        om = cosmo.cosmo.params.Omega_m
        ol = cosmo.cosmo.params.Omega_l
        x = a * (ol / om)**(1. / 3.)
        B0 = self._cmin(x) * self.cnorm
        B1 = self._imin(x) * self.inorm
        sig_p = B1 * sig
        Cc = 2.881 * ((sig_p / 1.257)**1.022 + 1) * np.exp(0.060 / sig_p**2)
        return B0 * Cc
