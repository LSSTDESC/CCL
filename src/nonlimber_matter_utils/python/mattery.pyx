"""
.. module:: matterlib
    :synopsis: Python wrapper around Matter module of CLASS
.. moduleauthor:: Nils Schöneberg <schoeneberg@physik.rwth-aachen.de>
"""
from math import exp,log
import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport *
import cython
cimport cython

ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i

# Import the .pxd containing definitions
from mattery cimport *

class MatterError(Exception):
    def __init__(self, message=""):
        self.message = message.decode() if isinstance(message,bytes) else message

    def __str__(self):
        return '\n\nError in Matter: ' + self.message

class MatterOutputError(MatterError):
    """
    Raised when matter failed during the output
    """
    pass


class MatterComputationError(MatterError):
    """
    Raised when matter failed during the computation
    """
    pass


cdef class Matter:
    """
    from matter import Matter
    """
    cpdef fft_real
    cpdef fft_imag
    cdef matters ma
    cpdef int computable

    # Called at the end of a run, to free memory
    def clean(self):
        free(self.ma.num_windows_per_cltp)
        free(self.ma.tw_min)
        free(self.ma.tw_max)
        free(self.ma.tw_sampling)
        free(self.ma.integrated_tw_sampling)
        free(self.ma.exp_integrated_tw_sampling)
        free(self.ma.tw_weights)
        free(self.ma.integrated_tw_weights)
        free(self.ma.ptw_sampling)
        free(self.ma.ptw_integrated_sampling)
        free(self.ma.tau_sampling)
        free(self.ma.sampled_sources)
        free(self.ma.k_sampling)
        free(self.ma.logk_sampling)
        for i in range(self.ma.radtp_size_total):
          free(self.ma.ptw_window[i])
        free(self.ma.ptw_window)
        free(self.ma.growth_factor_tau)
        #matter_free(&self.ma)

    # Called before any run, to set up the module
    def __init__(self,ma_verbose=0):
        """
        """
        self.computable = False
        self.ma.uses_intxi_logarithmic = 1
        self.ma.matter_verbose = ma_verbose

    # Called before any run, to set all relevant parameters
    # Some part of this can be done before any cosmology
    # Another part only when the cosmology is known
    # I didn't yet have the time to properly split those two
    # Since the time is only around ~0.05 seconds for this function,
    # it should be irrelevant
    def set(self,chi,kfac,pk,growth,lmax=50,**kwargs):
        """
        """
        # Non-integrated and integrated chi, number of windows and number of chi steps
        chi_ni,chi_i = chi
        ntr_ni = len(chi_ni)
        ntr_i = len(chi_i)
        nchi_ni = len(chi_ni[0])
        nchi_i = len(chi_i[0])
        # kernels and pk
        kfac_g,kfac_s = kfac
        a_pk,tau_pk, k_pk, pk = pk

        # Setting parameters of the matter module
        self.ma.has_cls = True
        self.ma.l_lss_max = lmax

        self.ma.has_unintegrated_windows = 1
        self.ma.has_integrated_windows = 1
        self.ma.uses_limber_approximation = 0
        self.ma.radtp_size_total = 2

        # Setting parameters of the matter module that can be set from the outside (overwritten as kwargs)
        self.ma.tau0 = 14000.
        self.ma.size_fft_cutoff = 100
        self.ma.tw_size = 50#25
        self.ma.integrated_tw_size = 150#75
        self.ma.t_size = 250 #150 in the file
        self.ma.t_spline_size = 60
        self.ma.uses_separability = 1
        self.ma.bias = 1.9
        self.ma.l_logstep = 1.5#1.12
        self.ma.l_linstep = 80#40
        self.ma.uses_bessel_store = 1
        internal_logchi_i_offset = 1e-10
        # Overwrite the parameters according to the kwargs
        for param in kwargs:
          if param == "size_fft_cutoff":
            self.ma.size_fft_cutoff = kwargs[param]
          elif param == "tau0":
            self.ma.tau0 = kwargs[param]
          elif param == "tw_size":
            self.ma.tw_size = kwargs[param]
          elif param == "integrated_tw_size":
            self.ma.integrated_tw_size = kwargs[param]
          elif param == "t_size":
            self.ma.t_size = kwargs[param]
          elif param == "t_spline_size":
            self.ma.t_spline_size = kwargs[param]
          elif param == "uses_separability":
            self.ma.uses_separability = kwargs[param]
          elif param == "bias":
            self.ma.bias = kwargs[param]
          elif param == "l_logstep":
            self.ma.l_logstep = kwargs[param]
          elif param == "l_linstep":
            self.ma.l_linstep = kwargs[param]
          elif param == "uses_bessel_store":
            self.ma.uses_bessel_store = kwargs[param]
          elif param == "internal_logchi_i_offset":
            internal_logchi_i_offset = kwargs[param]
          else:
            print("Unrecognized parameter '%s'='%s'"%(param,kwargs[param]))

        # Setting parameters derived from the chi windows
        self.ma.size_fft_input = len(k_pk)
        self.ma.num_windows = max(ntr_ni,ntr_i)
        self.ma.non_diag = self.ma.num_windows -1
        self.ma.num_windows_per_cltp = <int*>malloc(sizeof(int)*2) # 2 cltypes (nonint, int)
        self.ma.num_windows_per_cltp[0] = ntr_ni
        self.ma.num_windows_per_cltp[1] = ntr_i
        assert(self.ma.size_fft_cutoff < self.ma.size_fft_input)
        ntw_ni = self.ma.tw_size
        ntw_i = self.ma.integrated_tw_size

        # Allocating all the samplings
        self.ma.tw_min = <double*>malloc(sizeof(double)*(ntr_ni+ntr_i))
        self.ma.tw_max = <double*>malloc(sizeof(double)*(ntr_ni+ntr_i))
        self.ma.tw_sampling = <double*>malloc(sizeof(double)*self.ma.num_windows*ntw_ni)
        self.ma.integrated_tw_sampling = <double*>malloc(sizeof(double)*self.ma.num_windows*ntw_i)
        self.ma.exp_integrated_tw_sampling = <double*>malloc(sizeof(double)*self.ma.num_windows*ntw_i)
        self.ma.tw_weights = <double*>malloc(sizeof(double)*self.ma.num_windows*ntw_ni)
        self.ma.integrated_tw_weights = <double*>malloc(sizeof(double)*self.ma.num_windows*ntw_i)
        self.ma.ptw_sampling = <double*>malloc(sizeof(double)*self.ma.num_windows*nchi_ni)
        self.ma.ptw_integrated_sampling = <double*>malloc(sizeof(double)*self.ma.num_windows*nchi_i)
        self.ma.ptw_size = nchi_ni
        self.ma.ptw_integrated_size = nchi_i

        # Setting the relevant tau_window = tw samplings
        tw_ni = [self.ma.tau0-np.linspace(chi_ni[nwd][0],chi_ni[nwd][-1],num=ntw_ni)[::-1] for nwd in range(ntr_ni)]
        tw_i = [self.ma.tau0-np.linspace(chi_i[nwd][0],chi_i[nwd][-1],num=ntw_i)[::-1] for nwd in range(ntr_i)]
        logtw_i = [np.linspace(np.log(chi_i[nwd][0]+float(internal_logchi_i_offset)),np.log(chi_i[nwd][-1]),num=ntw_i) for nwd in range(ntr_i)]
        # Setting the corresponding weights
        tw_ni_weights = [self.weights(tw_ni[i]) for i in range(ntr_ni)]
        tw_i_weights = [self.weights(tw_i[i]) for i in range(ntr_i)]
        logtw_i_weights = [self.weights(logtw_i[i]) for i in range(ntr_i)]

        # Setting xmin to 1e-10
        xmin = 1e-10
        # The windows can be possibly cut earlier, but we don't use it here
        uses_lensing_reduction = False
        if uses_lensing_reduction:
          nu_reduction = 0.5*self.ma.bias-3
          window_epsilon = 1e-6
          chimax = chi_i[nwd][-1]
          chi_epsilon = (0.5*chimax)*pow(window_epsilon,-1./nu_reduction)
          xmin = max(chi_epsilon,1e-10)

        # Actually setting the samplings
        for nwd in range(ntr_ni):
          for itw in range(ntw_ni):
            self.ma.tw_sampling[nwd*ntw_ni+itw] = tw_ni[nwd][itw]
            self.ma.tw_weights[nwd*ntw_ni+itw] = tw_ni_weights[nwd][itw]
          self.ma.tw_max[nwd]=tw_ni[nwd][-1]
          self.ma.tw_min[nwd]=tw_ni[nwd][0]
        for nwd in range(ntr_i):
          self.ma.tw_max[ntr_ni+nwd]=tw_i[nwd][-1]
          self.ma.tw_min[ntr_ni+nwd]=tw_i[nwd][0]
        for nwd in range(ntr_i):
          for itw in range(ntw_i):
            if self.ma.uses_intxi_logarithmic:
                self.ma.exp_integrated_tw_sampling[nwd*ntw_i+itw] = np.exp(logtw_i[nwd][itw])
                self.ma.integrated_tw_sampling[nwd*ntw_i+itw] = logtw_i[nwd][itw]
                self.ma.integrated_tw_weights[nwd*ntw_i+itw] = logtw_i_weights[nwd][itw]
            else:
                self.ma.integrated_tw_sampling[nwd*ntw_i+itw] = tw_i[nwd][itw]
                self.ma.integrated_tw_weights[nwd*ntw_i+itw] = tw_i_weights[nwd][itw]
        for nwd in range(ntr_i,ntr_ni): # TODO :: remove
          for itw in range(ntw_i):
            if self.ma.uses_intxi_logarithmic:
                self.ma.exp_integrated_tw_sampling[nwd*ntw_i+itw] = 0.
            self.ma.integrated_tw_sampling[nwd*ntw_i+itw] = 0.
            self.ma.integrated_tw_weights[nwd*ntw_i+itw] = 0.
        for nwd in range(ntr_ni):
          for ichi in range(nchi_ni):
            self.ma.ptw_sampling[nwd*nchi_ni+ichi] = self.ma.tau0-chi_ni[nwd][::-1][ichi]
        for nwd in range(ntr_i):
          for ichi in range(nchi_i):
            self.ma.ptw_integrated_sampling[nwd*nchi_i+ichi] = self.ma.tau0-chi_i[nwd][::-1][ichi]
        for nwd in range(ntr_i,ntr_ni):
          for ichi in range(nchi_i):
            self.ma.ptw_integrated_sampling[nwd*nchi_i+ichi] = 0.

        # Allocate and set the samplings related to the P(k)
        self.ma.tau_size = len(a_pk)
        self.ma.tau_sampling = <double*>malloc(sizeof(double)*len(a_pk))
        self.ma.sampled_sources = <double*>malloc(sizeof(double)*len(a_pk)*len(k_pk))
        for ia in range(len(a_pk)):
          self.ma.tau_sampling[ia] = self.ma.tau0-tau_pk[ia]
          for ik in range(len(k_pk)):
            self.ma.sampled_sources[ik*len(a_pk)+ia] = np.sqrt(pk[ia,ik])
        self.ma.k_sampling = <double*>malloc(sizeof(double)*len(k_pk))
        self.ma.logk_sampling = <double*>malloc(sizeof(double)*len(k_pk))
        for ik in range(len(k_pk)):
          self.ma.k_sampling[ik] = k_pk[ik]
          self.ma.logk_sampling[ik] = np.log(k_pk[ik])
        self.ma.deltalogk = self.ma.logk_sampling[len(k_pk)-1]-self.ma.logk_sampling[0]

        # Allocate and set the samples related to the window function (kernel)
        self.ma.ptw_window = <double**>malloc(sizeof(double*)*self.ma.radtp_size_total)
        for i in range(self.ma.radtp_size_total):
          size = (nchi_i if self.matter_is_integrated(i) else nchi_ni)
          self.ma.ptw_window[i] = <double*>malloc(sizeof(double)*size*self.ma.num_windows)
          if self.matter_is_integrated(i):
            for nwd in range(ntr_i):
              for ichi in range(nchi_i):
                self.ma.ptw_window[i][nwd*nchi_i+ichi] = kfac_s[nwd][::-1][ichi]
            for nwd in range(ntr_i,ntr_ni):
              for ichi in range(nchi_i):
                self.ma.ptw_window[i][nwd*nchi_i+ichi] = 0.
          else:
            for nwd in range(ntr_ni):
              for ichi in range(nchi_ni):
                self.ma.ptw_window[i][nwd*nchi_ni+ichi] = kfac_g[nwd][::-1][ichi]

        # Allocate and set the growth factor
        self.ma.growth_factor_tau = <double*>malloc(sizeof(double)*self.ma.tau_size)
        for ia in range(len(a_pk)):
          self.ma.growth_factor_tau[ia] = growth[ia]

        # Set to computable
        self.computable = True

    # A small helper function to get the trapezoidal integration weights for a given (equally spaced) sampling
    def weights(self,x):

        warray = np.full((len(x),),(x[-1]-x[0])/(len(x)-1))
        warray[0]*=0.5
        warray[len(x)-1]*=0.5

        return warray

    # A function to get whether a certain index is integrated or not
    def matter_is_integrated(self,radtp):
      if radtp == 0:
        return False
      else:
        return True

    # A function to do the actual computation
    def compute(self):
        """
        """
        if not self.computable:
          return False
        cdef ErrorMsg errmsg

        # Initialize (and run) the matter module
        if matter_init(&(self.ma)) == _FAILURE_:
            raise MatterComputationError(self.ma.error_message)

        # Retrieve the fft coefficients
        # We don't need this in the end, but it can be useful for testing
        self.fft_real = np.empty((self.ma.size_fft_cutoff,))
        self.fft_imag = np.empty((self.ma.size_fft_cutoff,))
        for ifft in range(self.ma.size_fft_cutoff):
          self.fft_real[ifft] = self.ma.fft_real[ifft]
          self.fft_imag[ifft] = self.ma.fft_imag[ifft]
        return

    # A function to return the fft coefficents stored above
    def get_fft(self):
        return (self.fft_real,self.fft_imag)

    # A function to return the Cl's of different types
    def matter_cl(self, ells, nofail=False):
        def index_symmetric_matrix(a,b,N):
            if(a <= b):
                return b+N*a-(a*(a+1))//2
            else:
                return a+N*b-(b*(b+1))//2
        """
        matter_density_cl(ells, nofail=False)

        Return a dictionary of the primary number count/shear C_l for the matter structure

        Parameters
        ----------
        ells : array
            Define the l array for which the C_l will be returned
        nofail: bool, optional
            Check and enforce the computation of the lensing module beforehand

        Returns
        -------
        cl : numpy array of numpy.ndarrays
            Array that contains the list (in this order) of self correlation of
            1st bin, then successive correlations (set by non_diagonal) to the
            following bins, then self correlation of 2nd bin, etc. The array
            starts at index_ct_dd.
        """
        cdef int nl = len(ells)
        # Allocate array of Cl's to be used for interfacing with the C code
        cdef double **dcl = <double**> calloc(self.ma.cltp_grid_size,sizeof(double*))
        for index_cltp_grid in range(self.ma.cltp_grid_size):
            dcl[index_cltp_grid] = <double*> calloc(self.ma.window_size[index_cltp_grid]*nl, sizeof(double))

        lmaxR = self.ma.l_lss_max

        # Check that the get matter_cl request is valid
        if (not self.ma.has_cltp_nc) and (not self.ma.has_cltp_sh):
            raise MatterOutputError("No density Cl computed with matters struct")
        if ells[-1] > lmaxR:
            if nofail:
                self._pars_check("l_max_lss",ells[-1])
                self._pars_check("output",'nCl')
                self.compute()
            else:
                raise MatterOutputError("Can only compute up to lmax=%d, but l requested was %d"%(lmaxR,ells[-1]))

        # Which types to compute?
        cl = {}
        spectra = []
        if self.ma.has_cltp_nc:
            spectra.append('dd')
            if self.ma.has_cltp_sh:
                spectra.append('dl')
        if self.ma.has_cltp_sh:
            spectra.append('ll')

        # Set the cl type grid indices
        indices_cltp_grid = {}
        if self.ma.has_cltp_nc:
            index_cltp_grid = index_symmetric_matrix(self.ma.cltp_index_nc,self.ma.cltp_index_nc,self.ma.cltp_size)
            indices_cltp_grid['dd']=index_cltp_grid
            if self.ma.has_cltp_sh:
                index_cltp_grid = index_symmetric_matrix(self.ma.cltp_index_nc,self.ma.cltp_index_sh,self.ma.cltp_size)
                indices_cltp_grid['dl']=index_cltp_grid
        if self.ma.has_cltp_sh:
            index_cltp_grid = index_symmetric_matrix(self.ma.cltp_index_sh,self.ma.cltp_index_sh,self.ma.cltp_size)
            indices_cltp_grid['ll']=index_cltp_grid

        # Get the C'ls from the code
        ell_view = <double*>malloc(nl*sizeof(double))
        for il in range(nl):
          ell_view[il] = ells[il]
        if matter_cl_at_l(&self.ma, ell_view, nl, dcl) == _FAILURE_:
            raise MatterOutputError(self.ma.error_message)

        # Transpose the Cl's from the temporary dcl array into the cl output array
        for elem in spectra:
            cl[elem] = {}
            wd_size_elem = self.ma.window_size[indices_cltp_grid[elem]]
            index = 0
            for index_wd1 in range(self.ma.num_windows):
              for index_wd2 in range(self.ma.window_index_start[indices_cltp_grid[elem]][index_wd1],self.ma.window_index_end[indices_cltp_grid[elem]][index_wd1]+1):
                cl[elem][(index_wd1,index_wd2)] = np.zeros(nl,dtype="float64")
                for il,ellval in enumerate(ells):
                  cl[elem][(index_wd1,index_wd2)][il] = dcl[indices_cltp_grid[elem]][index+il*wd_size_elem]
                index+=1

        cl['ell'] = ells

        # Free temporary arrays
        for index_cltp_grid in range(self.ma.cltp_grid_size):
            free(dcl[index_cltp_grid])
        free(dcl)

        # Return the cl's
        return cl
