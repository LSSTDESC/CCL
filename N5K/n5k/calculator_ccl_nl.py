import numpy as np
import pyccl as ccl
from .calculator_ccl import N5KCalculatorCCL


class N5KCalculatorCCLNonLimber(N5KCalculatorCCL):
    name = 'CCLNonLimber'

    def _get_cl(self, t1, t2, ls, dchi, lnl):
        return ccl.angular_cl(self.cosmo, t1, t2, ls,
                              l_limber=lnl,
                              limber_integration_method='spline',
                              dchi_nonlimber=dchi)

    def run(self):
        # Compute power spectra
        ls = self.get_ells()
        # Radial interval (in Mpc)
        dchi = self.config.get('d_chi', 5.)
        # Radial interval (in Mpc)
        l_nonlimber = self.config.get('l_nonlimber', 100)

        self.cls_gg = []
        self.cls_gs = []
        self.cls_ss = []
        for i1, t1 in enumerate(self.t_g):
            for i2, t2 in enumerate(self.t_g[i1:]):
                print(i1, i2)
                self.cls_gg.append(self._get_cl(t1, t2, ls, dchi,
                                                l_nonlimber))
            for i2, t2 in enumerate(self.t_s):
                print(i1, i2)
                self.cls_gs.append(self._get_cl(t1, t2, ls, dchi,
                                                l_nonlimber))
        for i1, t1 in enumerate(self.t_s):
            for i2, t2 in enumerate(self.t_s[i1:]):
                print(i1, i2)
                self.cls_ss.append(self._get_cl(t1, t2, ls, dchi,
                                                l_nonlimber))
        self.cls_gg = np.array(self.cls_gg)
        self.cls_gs = np.array(self.cls_gs)
        self.cls_ss = np.array(self.cls_ss)
