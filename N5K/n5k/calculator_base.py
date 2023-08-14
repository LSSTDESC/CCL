import numpy as np


class N5KCalculatorBase(object):
    name = 'Base'
    needed_fields = ['output_prefix']

    def __init__(self, fname_config):
        if isinstance(fname_config, dict):
            self.config = fname_config
        else:
            import yaml

            with open(fname_config) as f:
                self.config = yaml.safe_load(f)

        self.nb_g = 10
        self.nb_s = 5
        if 'select_cl' in self.config:
            self.nb_g = len(self.config['select_cl'])
        if 'select_sh' in self.config:
            self.nb_s = len(self.config['select_sh'])

        self._check_config_sanity()

    def _check_config_sanity(self):
        for name in self.needed_fields:
            if not self.config.get(name):
                raise ValueError("You must provide %s"%(name))

    def get_pk(self):
        return np.load('input/pk.npz')

    def get_background(self):
        return np.load('input/background.npz')

    def get_cosmological_parameters(self):
        return {'Omega_m': 0.3156,
                'Omega_b': 0.0492,
                'w0': -1.0,
                'h': 0.6727,
                'A_s': 2.12107E-9,
                'n_s': 0.9645}

    def get_tracer_parameters(self):
        # Per-bin galaxy bias
        b_g = np.array([1.376695, 1.451179, 1.528404,
                        1.607983, 1.689579, 1.772899,
                        1.857700, 1.943754, 2.030887,
                        2.118943])
        return {'b_g': b_g}

    def get_tracer_dndzs(self):
        filename = self.config.get('dndz_file', 'input/dNdzs_fullwidth.npz')
        dNdz_file = np.load(filename)
        z_sh = dNdz_file['z_sh']
        dNdz_sh = dNdz_file['dNdz_sh']
        z_cl = dNdz_file['z_cl']
        dNdz_cl = dNdz_file['dNdz_cl']
        if 'select_cl' in self.config:
            dNdz_cl = np.array([dNdz_cl[:, i] for i in self.config['select_cl']]).T
        if 'select_sh' in self.config:
            dNdz_sh = np.array([dNdz_sh[:, i] for i in self.config['select_sh']]).T
        return {'z_sh': z_sh, 'dNdz_sh': dNdz_sh,
                'z_cl': z_cl, 'dNdz_cl': dNdz_cl}

    def get_noise_biases(self):
        from scipy.integrate import simps

        # Lens sample: 40 gals/arcmin^2
        ndens_c = 40.
        # Source sample: 27 gals/arcmin^2
        ndens_s = 27.
        # Ellipticity scatter per component
        e_rms = 0.28
        
        ndic = self.get_tracer_dndzs()
        nc_ints = np.array([simps(n, x=ndic['z_cl'])
                            for n in ndic['dNdz_cl'].T])
        ns_ints = np.array([simps(n, x=ndic['z_sh'])
                            for n in ndic['dNdz_sh'].T])
        nc_ints *= ndens_c / np.sum(nc_ints)
        ns_ints *= ndens_s / np.sum(ns_ints)
        tosrad = (180*60/np.pi)**2
        nl_cl = 1./(nc_ints*tosrad)
        nl_sh = e_rms**2/(ns_ints*tosrad)
        return nl_cl, nl_sh

    def get_tracer_kernels(self):
        filename = self.config.get('kernel_file', 'input/kernels_fullwidth.npz')
        d = np.load(filename)
        kernels_cl = d['kernels_cl']
        kernels_sh = d['kernels_sh']
        if 'select_cl' in self.config:
            kernels_cl = np.array([kernels_cl[i, :] for i in self.config['select_cl']])
        if 'select_sh' in self.config:
            kernels_sh = np.array([kernels_sh[i, :] for i in self.config['select_sh']])
        return {'z_cl': d['z_cl'],
                'chi_cl': d['chi_cl'],
                'kernels_cl': kernels_cl,
                'z_sh': d['z_sh'],
                'chi_sh': d['chi_sh'],
                'kernels_sh': kernels_sh}

    def get_ells(self):
        return np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)

    def get_nmodes_fullsky(self):
        """ Returns the number of modes in each ell bin"""
        ls = self.get_ells()
        nmodes = list(ls[1:]**2-ls[:-1]**2)
        lp = ls[-1]**2/ls[-2]
        nmodes.append(lp**2-ls[-1]**2)
        return np.array(nmodes)*0.5

    def get_num_cls(self):
        ngg = (self.nb_g * (self.nb_g + 1)) // 2
        nss = (self.nb_s * (self.nb_s + 1)) // 2
        ngs = self.nb_g * self.nb_s
        return ngg, ngs, nss

    def write_output(self):
        ls = self.get_ells()
        nl = len(ls)
        ngg, ngs, nss = self.get_num_cls()
        if self.cls_gg.shape != (ngg, nl):
            raise ValueError("Incorrect G-G spectra shape")
        if self.cls_gs.shape != (ngs, nl):
            raise ValueError("Incorrect G-S spectra shape")
        if self.cls_ss.shape != (nss, nl):
            raise ValueError("Incorrect S-S spectra shape")

        np.savez(self.config['output_prefix'] + '_clgg.npz',
                 ls=ls, cls=self.cls_gg)
        np.savez(self.config['output_prefix'] + '_clgs.npz',
                 ls=ls, cls=self.cls_gs)
        np.savez(self.config['output_prefix'] + '_clss.npz',
                 ls=ls, cls=self.cls_ss)

    def teardown(self):
        pass

    def setup(self):
        pass

    def run(self):
        nl = len(self.get_ells())
        ngg, ngs, nss = self.get_num_cls()
        self.cls_gg = np.zeros((ngg, nl))
        self.cls_gs = np.zeros((ngs, nl))
        self.cls_ss = np.zeros((nss, nl))
