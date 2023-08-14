import numpy as np
import pyccl as ccl
from .calculator_base import N5KCalculatorBase
import matterlib
from scipy.interpolate import CubicSpline as interp
import matplotlib.pyplot as plt

# @author : Nils Schöneberg
# Although I wrote this whole code by myself, in documentation I often write 'we'. Sorry about that.

class N5KCalculatorMATTER(N5KCalculatorBase):
    name = 'MATTER'

    def setup(self):
        # RELEVANT PRECISION PARAMTERS
        verbosity = self.config['verbosity']
        sfftcutoff = self.config['size_FFTlog']
        stw = self.config['size_chi_window_array']
        sitw = self.config['size_chi_window_array_integrated']
        st = self.config['size_prep_t_array']
        st_spline = self.config['size_t_array']
        l_logstep = self.config['l_logstep']
        l_linstep = self.config['l_linstep']
        self.lmax = self.config['lmax']
        seperability = self.config['seperable']
        kmin = float(self.config['k_min'])
        self.Nk_fft = self.config['size_prep_FFTlog']
        Nchi_nonintegrated = self.config['size_prep_chi_window_array']
        Nchi_integrated = self.config['size_prep_chi_window_array_integrated']
        internal_logchi_i_offset = self.config['internal_logchi_i_offset']

        # INITIALIZE COSMOLOGY
        par = self.get_cosmological_parameters()
        self.cosmo = ccl.Cosmology(Omega_c=par['Omega_m']-par['Omega_b'],
                                   Omega_b=par['Omega_b'],
                                   h=par['h'], n_s=par['n_s'],
                                   A_s=par['A_s'], w0=par['w0'])
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

        # INITIALIZE TRACERS
        #  -> Currently, we only allow this to be done from the kernel file,
        #     but it would be (relatively) easy to extend this also to RSD,
        #     GR corrections, lensing of the number counts, etc.
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

        # PREPROCESSING
        # -> Now we perform (!)preprocessing(!) steps relevant to the 'matter (FFTlog)' calculator
        #    All of these steps are considered to be possible before any calculation
        #    and could have instead been done be slightly rewriting input/generate_inputs.py
        #    or alternatively changing the input/pk.npz and input/kernels_fullwidth.npz files
        #    Since, however, we wanted to leave the directory structure intact,
        #    we do it here in a pre-processing step instead

        # 1) Define some shorthand notation
        Ntg = len(self.t_g)
        Nts = len(self.t_s)

        # 2) Find out what is the minimum and maximum relevant value of chi for each seperate kernel window function
        #  -> Note, that this step would be unnecessary, if the files would be storing
        #     only the relevant nonzero contributions
        #  -> We iterate through each window function, sample it on a very broad grid, and check what is the maximum value
        threshold = 1e-50
        chi_test = ker['chi_cl']
        self.chi_g_mins,self.chi_g_maxs = np.empty((2,Ntg),dtype="float64")
        self.chi_s_mins,self.chi_s_maxs = np.zeros((2,Nts),dtype="float64")
        for i,tg in enumerate(self.t_g):
           # Get the kernel
           tg_test = tg.get_kernel(chi_test)
           # Get the maximum value
           maxtg = np.max(tg_test)
           # Check where the kernel is nonzero = bigger than it's maximum times some small threshold (1e-10)
           mask = tg_test>threshold*maxtg
           self.chi_g_maxs[i] = chi_test[len(mask[0])-np.argmax(mask[0][::-1])-1]
           self.chi_g_mins[i] = chi_test[np.argmax(mask[0])]
        chi_test = ker['chi_sh']
        for i,ts in enumerate(self.t_s):
           # Get the kernel
           ts_test = ts.get_kernel(chi_test)
           # Get the maximum value
           maxts = np.max(ts_test)
           # Check where the kernel is nonzero = bigger than it's maximum times some small threshold (1e-10)
           mask = ts_test>threshold*maxts
           self.chi_s_maxs[i] = chi_test[len(mask[0])-np.argmax(mask[0][::-1])-1]
           imin = np.argmax(mask)
           # If imin==0, this means that the shear window function has support up until chi=z=0
           self.chi_s_mins[i] = 1.#(chi_test[imin] if imin>1 else 0.)

        # Now the minima and maxima in chi of the window functions have been found

        # 3) Define corresponding chi sampling
        #  -> Note, that for this we purposefully re-define the chi sampling for each window seperately
        #  -> Note, that this step needn't be done if the kernel file would be re-arranged
        self.chi_nonintegrated = [np.linspace(self.chi_g_mins[i],self.chi_g_maxs[i],num=Nchi_nonintegrated) for i in range(Ntg)]
        self.chi_integrated = [np.linspace(np.min(self.chi_s_mins[i]),np.max(self.chi_s_maxs[i]),num=Nchi_integrated) for i in range(Nts)]

        # 4) Get the Kernel and Transfer at this chi sampling
        #  -> Note, that this step would also be unnecessary for a different layout of the kernel file
        #  -> A 'rearranged' kernel file could simply contain self.chi_nonintegrated, self.kerfac_g, self.chi_integrated, self.kerfac_s
        self.kerfac_g = np.zeros((Ntg,Nchi_nonintegrated))
        for i in range(Ntg):
          kern = self.t_g[i].get_kernel(self.chi_nonintegrated[i])
          trans = self.t_g[i].get_transfer(0.,ccl.scale_factor_of_chi(self.cosmo,self.chi_nonintegrated[i]))

          for itr in range(kern.shape[0]):
            self.kerfac_g[i] += kern[itr]*trans[itr]

        self.kerfac_s = np.zeros((Nts,Nchi_integrated))
        for i in range(Nts):
          kern = self.t_s[i].get_kernel(self.chi_integrated[i])
          trans = self.t_s[i].get_transfer(0.,ccl.scale_factor_of_chi(self.cosmo,self.chi_integrated[i]))

          for itr in range(kern.shape[0]):
            self.kerfac_s[i][1:] += (kern[itr][1:]*trans[itr][1:]/self.chi_integrated[i][1:]**2) # Ill defined for the 0th index, keep it as 0

        # Shorthand notations:
        power = dpk['pk_nl'][::-1]
        Na_pk = len(power)

        # 5) Get the growth factor and pass it as well (sampled on same scale factor grid as the P(k))
        #   -> This could be in a file similar to the kernels_fullwidth.npz or pk.npz
        #      Since below we derive it entirely from the file
        #      it is data that could also be easily provided alongside it
        self.a_pk = a
        pk_growth = ccl.growth_factor(self.cosmo, self.a_pk)
        self.growth = pk_growth

        # 6) Multiply the growth factor into the kernel functions
        # !! Whether this part should be done in the setup or in the run method is a bit debatable. It's so fast it shouldn't matter though! (For convenience, we do it here)
        growth_func = interp(self.a_pk,pk_growth)
        for i in range(Ntg):
          self.kerfac_g[i] *= growth_func(ccl.scale_factor_of_chi(self.cosmo,self.chi_nonintegrated[i]))
        for i in range(Nts):
          self.kerfac_s[i] *= growth_func(ccl.scale_factor_of_chi(self.cosmo,self.chi_integrated[i]))

        # 7) The kmin of the provided file is a bit too high. Here we extrapolate using k^(n_s) to reach lower k values (down to kmin = 1e-7/Mpc)
        #   -> This would also not need to be done if the file was a bit more expansive in its k-range
        new_k_min = float(self.config['k_min'])#1e-7
        Nk_small = int(np.log10(dpk['k'][0]/new_k_min)/np.log10(dpk['k'][1]/dpk['k'][0])+1)
        assert(Nk_small > 10)
        ksmall = np.geomspace(new_k_min,dpk['k'][0],endpoint=False,num=Nk_small)
        k_all = np.concatenate([ksmall,dpk['k']])

        # 8) Now we do some transformations of the P(k,z) file, getting some derived quantities
        #   -> Again, this could be done in a pre-processing step, generating a different input/pk.npz file.
        self.chi_pk = ccl.comoving_radial_distance(self.cosmo,self.a_pk)
        self.k_pk = np.geomspace(kmin,dpk['k'][-1],num=self.Nk_fft)
        self.pk = np.empty((Na_pk,self.Nk_fft))
        self.deltaksq = np.empty((Na_pk,self.Nk_fft))
        for i in range(Na_pk):
          pk_all = np.concatenate([(ksmall/dpk['k'][0])**(par['n_s'])*power[i][0],power[i]])
          self.pk[i] = interp(k_all,pk_all)(self.k_pk)
          # Special note : we want Delta(k,z)=sqrt(P(k,z)*k^3/(2*pi^2)) instead of P(k) for the matterlib
          self.deltaksq[i] = interp(k_all,pk_all)(self.k_pk)*self.k_pk**3/(2.*np.pi**2)


        # SETTING UP THE MATTERLIB
        #  -> Finally, we can set up the python wrapper of matter.c, which is called 'matterlib'.
        # 1) Setup a matterlib object, giving it a desired verbosity
        #  -> This step does basically nothing, except for setting a few flags
        self.ma = matterlib.Matter(ma_verbose=verbosity)

        # PASS INPUTS TO THE MATTERLIB
        # !! This step is really debatable, whether it should be here or in the run method
        #  -> It is a combination of ALLOCATING space for the C code, which is completely independent of cosmology,
        #     passing cosmology independent arrays, like the kernel functions
        #     and passing cosmology-dependent things, like the growth factor
        #  -> We took the liberty to leave it here. It takes only around 0.05 seconds, so it should be irrelevant for timing tests
        #     If you are very concerned about it, please feel free to move it into the run method
        self.ma.set(
          (self.chi_nonintegrated,self.chi_integrated),
          (self.kerfac_g,self.kerfac_s),(self.a_pk,self.chi_pk,self.k_pk,self.deltaksq),
          self.growth,
          lmax=self.lmax, uses_separability=seperability,
          size_fft_cutoff=sfftcutoff,tw_size=stw,integrated_tw_size=sitw,
          l_logstep=l_logstep, l_linstep = l_linstep,
          t_size = st, t_spline_size = st_spline,
          internal_logchi_i_offset = internal_logchi_i_offset
          )

        all_ells = self.get_ells()
        self.ell_matter = all_ells[all_ells<=self.lmax]
        self.ell_limber = all_ells[all_ells>self.lmax]

    def run(self):
        # EXECUTE THE MATTERLIB
        # !! This is where most of the computational time is actually spent, especially in the loops doing the integration.
        # !! There is some minor ~0.05 second part of reading a Bessel Integral file, which could be done in the setup. We didn't worry about that yet,
        #    but it could be optimized for some final release version
        # Compute power spectra
        self.ma.compute()

        # GET RESULTS
        # Get the Cl's for the given l array
        cls_matter = self.ma.matter_cl(self.ell_matter)

        # Store the results in the Cl's_xy array (this is equivalent to the part in calculator_ccl.py, just with different notation)
        self.cls_gg = []
        self.cls_gs = []
        self.cls_ss = []
        for i1 in range(len(self.t_g)):
          for i2 in range(i1,len(self.t_g)):
            cls_m = cls_matter["dd"][(i1,i2)]
            cls_l = ccl.angular_cl(self.cosmo, self.t_g[i1], self.t_g[i2], self.ell_limber, limber_integration_method='qag_quad')
            self.cls_gg.append(np.concatenate([cls_m,cls_l]))
          for i2 in range(len(self.t_s)):
            cls_m = cls_matter["dl"][(i1,i2)]
            cls_l = ccl.angular_cl(self.cosmo, self.t_g[i1], self.t_s[i2], self.ell_limber, limber_integration_method='qag_quad')
            self.cls_gs.append(np.concatenate([cls_m,cls_l]))
        for i1 in range(len(self.t_s)):
          for i2 in range(i1,len(self.t_s)):
            cls_m = cls_matter["ll"][(i1,i2)]
            cls_l = ccl.angular_cl(self.cosmo, self.t_s[i1], self.t_s[i2], self.ell_limber, limber_integration_method='qag_quad')
            self.cls_ss.append(np.concatenate([cls_m,cls_l]))
        self.cls_gg = np.array(self.cls_gg)
        self.cls_gs = np.array(self.cls_gs)
        self.cls_ss = np.array(self.cls_ss)

        return

    def teardown(self):
        self.ma.clean()
        

