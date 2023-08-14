import numpy as np
from .calculator_base import N5KCalculatorBase

import levinpower

class N5KCalculatorLevin(N5KCalculatorBase):
    name = 'Levin'

    def setup(self):
        # Initialize cosmology
        pk = self.get_pk()
        kernels = self.get_tracer_kernels()
        background = self.get_background()

        number_count = kernels["kernels_cl"].shape[0]

        precompute_splines = self.config.get('precompute_splines', False)
        ell_max_non_Limber = self.config.get('ell_max_non_Limber', 95)
        ell_max_ext_Limber = self.config.get('ell_max_ext_Limber', 1000)

        extra_kwargs = {k: self.config[k]
                        for k in ["tol_rel",
                                  "limber_tolerance",
                                  "min_interval",
                                  "maximum_number_subintervals",
                                  "n_collocation"]
                        if k in self.config}

        ell = self.get_ells().astype(int)
        self.levin_calculator = levinpower.LevinPower(
                          ell,
                          number_count,
                          background["z"], background["chi"],
                          kernels["chi_cl"],
                          np.concatenate((kernels["kernels_cl"].T,
                                          kernels["kernels_sh"].T), axis=1),
                          pk["k"], pk["z"],
                          pk["pk_lin"].flatten(),
                          pk["pk_nl"].flatten(),
                          precompute_splines=precompute_splines,
                          ell_max_non_Limber=ell_max_non_Limber,
                          ell_max_ext_Limber=ell_max_ext_Limber,
                          **extra_kwargs)

    def run(self):
        # Compute power spectra
        parallelize_ell = self.config.get('parallelize_ell', False)
        self.cls_gg, self.cls_gs, self.cls_ss = \
            self.levin_calculator.compute_C_ells(parallelize_ell)

        self.cls_gg = np.array(self.cls_gg)
        self.cls_gs = np.array(self.cls_gs)
        self.cls_ss = np.array(self.cls_ss)
