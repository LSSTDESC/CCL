import numpy as np
import pyccl as ccl
from .calculator_base import N5KCalculatorBase


class N5KCalculatorCCL(N5KCalculatorBase):
    name = 'CCL'

    def setup(self):
        # Initialize cosmology
        par = self.get_cosmological_parameters()
        dpk = self.get_pk()
        a = 1./(1+dpk['z'][::-1])
        self.cosmo = ccl.CosmologyCalculator(Omega_c=par['Omega_m']-par['Omega_b'],
                                             Omega_b=par['Omega_b'],
                                             h=par['h'], n_s=par['n_s'],
                                             A_s=par['A_s'], w0=par['w0'],
                                             pk_linear={'a': a,
                                                        'k': dpk['k'],
                                                        'delta_matter:delta_matter': dpk['pk_lin'][::-1][:]},
                                             pk_nonlin={'a': a,
                                                        'k': dpk['k'],
                                                        'delta_matter:delta_matter': dpk['pk_nl'][::-1][:]})

        # Initialize tracers
        if self.config.get('tracers_from_kernels', False):
            tpar = self.get_tracer_parameters()
            ker = self.get_tracer_kernels()
            a_g = 1./(1+ker['z_cl'][::-1])
            self.t_g = []
            for k in ker['kernels_cl']:
                t = ccl.Tracer()
                barr = np.ones_like(a_g)
                t.add_tracer(self.cosmo,
                             (ker['chi_cl'], k),
                             transfer_a=(a_g, barr))
                self.t_g.append(t)
            self.t_s = []
            for k in ker['kernels_sh']:
                t = ccl.Tracer()
                t.add_tracer(self.cosmo,
                             kernel=(ker['chi_sh'], k),
                             der_bessel=-1, der_angles=2)
                self.t_s.append(t)
        else:
            nzs = self.get_tracer_dndzs()
            tpar = self.get_tracer_parameters()
            z_g = nzs['z_cl']
            z_s = nzs['z_sh']
            self.t_g = [ccl.NumberCountsTracer(self.cosmo, True,
                                               (z_g, nzs['dNdz_cl'][:, ni]),
                                               bias=(z_g,
                                                     np.full(len(z_g), b)))
                        for ni, b in zip(range(0, 10),
                                         tpar['b_g'])]
            self.t_s = [ccl.WeakLensingTracer(self.cosmo,
                                              (z_s, nzs['dNdz_sh'][:, ni]),
                                              True)
                        for ni in range(0, 5)]

    def _get_cl(self, t1, t2, ls):
        return ccl.angular_cl(self.cosmo, t1, t2, ls,
                              limber_integration_method='spline')

    def run(self):
        # Compute power spectra
        ls = self.get_ells()

        self.cls_gg = []
        self.cls_gs = []
        self.cls_ss = []
        for i1, t1 in enumerate(self.t_g):
            for t2 in self.t_g[i1:]:
                self.cls_gg.append(self._get_cl(t1, t2, ls))
            for t2 in self.t_s:
                self.cls_gs.append(self._get_cl(t1, t2, ls))
        for i1, t1 in enumerate(self.t_s):
            for t2 in self.t_s[i1:]:
                self.cls_ss.append(self._get_cl(t1, t2, ls))
        self.cls_gg = np.array(self.cls_gg)
        self.cls_gs = np.array(self.cls_gs)
        self.cls_ss = np.array(self.cls_ss)
