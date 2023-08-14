import matplotlib.pyplot as plt
import numpy as np
from .calculator_base import N5KCalculatorBase
from .utils import n5k_calculator_from_name


class N5KCalculatorTester(N5KCalculatorBase):
    name = 'Tester'

    def setup(self):
        self.fsky = 0.4
        self.indices_gg = []
        self.indices_gs = []
        self.indices_ss = []

        for i1 in range(10):
            for i2 in range(i1, 10):
                self.indices_gg.append((i1, i2))
            for i2 in range(5):
                self.indices_gs.append((i1, i2))
        for i1 in range(5):
            for i2 in range(i1, 5):
                self.indices_ss.append((i1, i2))

        self.indices_full = []
        for i1, i2 in self.indices_gg:
            self.indices_full.append((i1, i2))
        for i1, i2 in self.indices_gs:
            self.indices_full.append((i1, i2+10))
        for i1, i2 in self.indices_ss:
            self.indices_full.append((i1+10, i2+10))

        self.run()

    def setup_alt(self, conf):
        # use this method to compare to a subset of bins from the fiducial case.
        # must define select_sh and select_cl

        sh_num = len(conf['select_sh'])
        cl_num = len(conf['select_cl'])

        self.fsky = 0.4
        self.indices_gg = []
        self.indices_gs = []
        self.indices_ss = []

        for i1 in range(cl_num):
            for i2 in range(i1, cl_num):
                self.indices_gg.append((i1, i2))
            for i2 in range(sh_num):
                self.indices_gs.append((i1, i2))
        for i1 in range(sh_num):
            for i2 in range(i1, sh_num):
                self.indices_ss.append((i1, i2))

        self.indices_full = []
        for i1, i2 in self.indices_gg:
            self.indices_full.append((i1, i2))
        for i1, i2 in self.indices_gs:
            self.indices_full.append((i1, i2+cl_num))
        for i1, i2 in self.indices_ss:
            self.indices_full.append((i1+cl_num, i2+cl_num))

        self.run_alt(conf)


    def _read_cls(self, prefix):
        d = np.load(prefix + '_clgg.npz')
        ls = d['ls']
        cls_gg = d['cls']
        d = np.load(prefix + '_clgs.npz')
        cls_gs = d['cls']
        d = np.load(prefix + '_clss.npz')
        cls_ss = d['cls']
        return ls, cls_gg, cls_gs, cls_ss

    def _cls_to_matrix(self, cls_gg, cls_gs, cls_ss):
        # Get C_ell matrix
        cls_mat = np.zeros([len(self.ls), 15, 15])
        for i, cl in enumerate(cls_gg):
            i1, i2 = self.indices_gg[i]
            cls_mat[:, i1, i2] = cl
            if i1 != i2:
                cls_mat[:, i2, i1] = cl
        for i, cl in enumerate(cls_gs):
            i1, i2 = self.indices_gs[i]
            cls_mat[:, i1, i2+10] = cl
            cls_mat[:, i2+10, i1] = cl
        for i, cl in enumerate(cls_ss):
            i1, i2 = self.indices_ss[i]
            cls_mat[:, i1+10, i2+10] = cl
            if i1 != i2:
                cls_mat[:, i2+10, i1+10] = cl
        return cls_mat

    def _cls_to_matrix_alt(self, cls_gg, cls_gs, cls_ss, conf):
        # Get C_ell matrix - use this version using a non fiducial clustering and shear bin setup.
        # must define select_sh and select_cl in conf

        cl_num = len(conf['select_cl'])
        sh_num = len(conf['select_sh'])
        tot_num = cl_num + sh_num

        cls_mat = np.zeros([len(self.ls), tot_num, tot_num])
        for i, cl in enumerate(cls_gg):
            i1, i2 = self.indices_gg[i]
            cls_mat[:, i1, i2] = cl
            if i1 != i2:
                cls_mat[:, i2, i1] = cl
        for i, cl in enumerate(cls_gs):
            i1, i2 = self.indices_gs[i]
            cls_mat[:, i1, i2+cl_num] = cl
            cls_mat[:, i2+cl_num, i1] = cl
        for i, cl in enumerate(cls_ss):
            i1, i2 = self.indices_ss[i]
            cls_mat[:, i1+cl_num, i2+cl_num] = cl
            if i1 != i2:
                cls_mat[:, i2+cl_num, i1+cl_num] = cl
        return cls_mat

    def compare(self, calculator_name, fname_config, plot_stuff=False):
        cal = n5k_calculator_from_name(calculator_name)(fname_config)
        cal.setup()
        cal.run()
        cls_gg = cal.cls_gg.copy()
        cls_gs = cal.cls_gs.copy()
        cls_ss = cal.cls_ss.copy()
        cal.teardown()

        # Compute chi2 of the difference
        cls_test = self._cls_to_matrix(cls_gg, cls_gs, cls_ss)
        nmodes = self.fsky*self.get_nmodes_fullsky()
        dcl = cls_test-self.cls_mat

        # DC_l * C_l^{-1}
        clicl = np.einsum('lik,lkj->lij', dcl, self.cls_imat)
        # Tr((DC*C^{-1})^2)
        fisher_l = np.einsum('lik,lki->l', clicl, clicl)
        # sqrt(Sum_ell of the above)
        sn = np.sqrt(np.sum(nmodes * fisher_l))
        # Gaussian errors
        cls_gg_err = []
        cls_gs_err = []
        cls_ss_err = []
        for i, (i1, i2) in enumerate(self.indices_gg):
            cl11 = self.cls_mat_wn[:, i1, i1]
            cl22 = self.cls_mat_wn[:, i2, i2]
            cl12 = self.cls_mat_wn[:, i1, i2]
            cls_gg_err.append(np.sqrt((cl11*cl22+cl12**2)/nmodes))

        for i, (i1, i2) in enumerate(self.indices_gs):
            cl11 = self.cls_mat_wn[:, i1, i1]
            cl22 = self.cls_mat_wn[:, i2+10, i2+10]
            cl12 = self.cls_mat_wn[:, i1, i2+10]
            cls_gs_err.append(np.sqrt((cl11*cl22+cl12**2)/nmodes))

        for i, (i1, i2) in enumerate(self.indices_ss):
            cl11 = self.cls_mat_wn[:, i1+10, i1+10]
            cl22 = self.cls_mat_wn[:, i2+10, i2+10]
            cl12 = self.cls_mat_wn[:, i1+10, i2+10]
            cls_ss_err.append(np.sqrt((cl11*cl22+cl12**2)/nmodes))

        cls_gg_err = np.array(cls_gg_err)
        cls_gs_err = np.array(cls_gs_err)
        cls_ss_err = np.array(cls_ss_err)

        # Save power spectra
        np.savez(self.config['output_prefix'] + '_comp_' +
                 calculator_name + '.npz',
                 ls=self.ls, cl_gg_bm=self.cls_gg,
                 cl_gs_bm=self.cls_gs, cl_ss_bm=self.cls_ss,
                 cl_gg=cls_gg, cl_gs=cls_gs, cl_ss=cls_ss,
                 cl_gg_err=cls_gg_err, cl_gs_err=cls_gs_err,
                 cl_ss_err=cls_ss_err,
                 sn=sn, sn_per_l=np.sqrt(fisher_l*nmodes))

        if plot_stuff:
            # Make plots
            def plot_cls(fname, cl1, cl2, el):
                plt.figure()
                plt.plot(self.ls, (cl1-cl2)/el)
                plt.xlabel(r'$\ell$', fontsize=16)
                plt.ylabel(r'$\Delta C_\ell/\sigma(C_\ell)$', fontsize=16)
                plt.xscale('log')
                plt.savefig(fname, bbox_inches='tight')
                plt.close()

            for i, (i1, i2) in enumerate(self.indices_gg):
                el = cls_gg_err[i]
                plot_cls(self.config['output_prefix'] + 'clcomp' +
                         calculator_name + '_g%d_g%d.png' % (i1, i2),
                         self.cls_gg[i], cls_gg[i], el)

            for i, (i1, i2) in enumerate(self.indices_gs):
                el = cls_gs_err[i]
                plot_cls(self.config['output_prefix'] + 'clcomp' +
                         calculator_name + '_g%d_s%d.png' % (i1, i2),
                         self.cls_gs[i], cls_gs[i], el)

            for i, (i1, i2) in enumerate(self.indices_ss):
                el = cls_ss_err[i]
                plot_cls(self.config['output_prefix'] + 'clcomp' +
                         calculator_name + '_s%d_s%d.png' % (i1, i2),
                         self.cls_ss[i], cls_ss[i], el)
        return sn


    def compare_alt(self, calculator_name, fname_config, plot_stuff=False):
        cal = n5k_calculator_from_name(calculator_name)(fname_config)
        cal.setup()
        cal.run()
        cls_gg = cal.cls_gg.copy()
        cls_gs = cal.cls_gs.copy()
        cls_ss = cal.cls_ss.copy()
        cal.teardown()

        cl_num = len(fname_config['select_cl'])
        sh_num = len(fname_config['select_sh'])
        tot_num = cl_num + sh_num

        # Compute chi2 of the difference
        cls_test = self._cls_to_matrix_alt(cls_gg, cls_gs, cls_ss, fname_config)
        nmodes = self.fsky*self.get_nmodes_fullsky()
        dcl = cls_test-self.cls_mat

        # DC_l * C_l^{-1}
        clicl = np.einsum('lik,lkj->lij', dcl, self.cls_imat)
        # Tr((DC*C^{-1})^2)
        fisher_l = np.einsum('lik,lki->l', clicl, clicl)
        # sqrt(Sum_ell of the above)
        sn = np.sqrt(np.sum(nmodes * fisher_l))
        # Gaussian errors
        cls_gg_err = []
        cls_gs_err = []
        cls_ss_err = []
        for i, (i1, i2) in enumerate(self.indices_gg):
            cl11 = self.cls_mat_wn[:, i1, i1]
            cl22 = self.cls_mat_wn[:, i2, i2]
            cl12 = self.cls_mat_wn[:, i1, i2]
            cls_gg_err.append(np.sqrt((cl11*cl22+cl12**2)/nmodes))

        for i, (i1, i2) in enumerate(self.indices_gs):
            cl11 = self.cls_mat_wn[:, i1, i1]
            cl22 = self.cls_mat_wn[:, i2+cl_num, i2+cl_num]
            cl12 = self.cls_mat_wn[:, i1, i2+cl_num]
            cls_gs_err.append(np.sqrt((cl11*cl22+cl12**2)/nmodes))

        for i, (i1, i2) in enumerate(self.indices_ss):
            cl11 = self.cls_mat_wn[:, i1+cl_num, i1+cl_num]
            cl22 = self.cls_mat_wn[:, i2+cl_num, i2+cl_num]
            cl12 = self.cls_mat_wn[:, i1+cl_num, i2+cl_num]
            cls_ss_err.append(np.sqrt((cl11*cl22+cl12**2)/nmodes))

        cls_gg_err = np.array(cls_gg_err)
        cls_gs_err = np.array(cls_gs_err)
        cls_ss_err = np.array(cls_ss_err)

        # Save power spectra
        np.savez(self.config['output_prefix'] + '_comp_' +
                 calculator_name + '.npz',
                 ls=self.ls, cl_gg_bm=self.cls_gg,
                 cl_gs_bm=self.cls_gs, cl_ss_bm=self.cls_ss,
                 cl_gg=cls_gg, cl_gs=cls_gs, cl_ss=cls_ss,
                 cl_gg_err=cls_gg_err, cl_gs_err=cls_gs_err,
                 cl_ss_err=cls_ss_err,
                 sn=sn, sn_per_l=np.sqrt(fisher_l*nmodes))

        if plot_stuff:
            # Make plots
            def plot_cls(fname, cl1, cl2, el):
                plt.figure()
                plt.plot(self.ls, (cl1-cl2)/el)
                plt.xlabel(r'$\ell$', fontsize=16)
                plt.ylabel(r'$\Delta C_\ell/\sigma(C_\ell)$', fontsize=16)
                plt.xscale('log')
                plt.savefig(fname, bbox_inches='tight')
                plt.close()

            for i, (i1, i2) in enumerate(self.indices_gg):
                el = cls_gg_err[i]
                plot_cls(self.config['output_prefix'] + 'clcomp' +
                         calculator_name + '_g%d_g%d.png' % (i1, i2),
                         self.cls_gg[i], cls_gg[i], el)

            for i, (i1, i2) in enumerate(self.indices_gs):
                el = cls_gs_err[i]
                plot_cls(self.config['output_prefix'] + 'clcomp' +
                         calculator_name + '_g%d_s%d.png' % (i1, i2),
                         self.cls_gs[i], cls_gs[i], el)

            for i, (i1, i2) in enumerate(self.indices_ss):
                el = cls_ss_err[i]
                plot_cls(self.config['output_prefix'] + 'clcomp' +
                         calculator_name + '_s%d_s%d.png' % (i1, i2),
                         self.cls_ss[i], cls_ss[i], el)
        return sn

    def run(self):
        np.set_printoptions(linewidth=300)
        # Compute power spectra
        self.ls, self.cls_gg, self.cls_gs, self.cls_ss = self._read_cls(self.config['benchmark_prefix'])
        self.cls_mat = self._cls_to_matrix(self.cls_gg,
                                           self.cls_gs,
                                           self.cls_ss)
        nl_cl, nl_sh = self.get_noise_biases()
        nls_mat = np.diag(np.concatenate((nl_cl, nl_sh)))
        self.cls_mat_wn = self.cls_mat + nls_mat[None, :, :]
        self.cls_imat = np.linalg.inv(self.cls_mat_wn)

    def run_alt(self, conf):
        np.set_printoptions(linewidth=300)
        # Compute power spectra - use this version if using a non-standard number of bins
        self.ls, self.cls_gg, self.cls_gs, self.cls_ss = self._read_cls(self.config['benchmark_prefix'])
        self.cls_mat = self._cls_to_matrix_alt(self.cls_gg,
                                           self.cls_gs,
                                           self.cls_ss, conf)
        nl_cl, nl_sh = self.get_noise_biases()
        nls_mat = np.diag(np.concatenate((nl_cl, nl_sh)))
        self.cls_mat_wn = self.cls_mat + nls_mat[None, :, :]
        self.cls_imat = np.linalg.inv(self.cls_mat_wn)
