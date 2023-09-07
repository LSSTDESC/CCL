# Bunch of declarations from C to python. The idea here is to define only the
# quantities that will be used, for input, output or intermediate manipulation,
# by the python wrapper. For instance, in the precision structure, the only
# item used here is its error message. That is why nothing more is defined from
# this structure. The rest is internal in Class.
# If, for whatever reason, you need an other, existing parameter from Class,
# remember to add it inside this cdef.

DEF _MAX_NUMBER_OF_K_FILES_ = 30
DEF _MAXTITLESTRINGLENGTH_ = 8000
DEF _FILENAMESIZE_ = 256
DEF _LINE_LENGTH_MAX_ = 1024

cdef extern from "matter.h":

    ctypedef char FileArg[40]

    ctypedef char* ErrorMsg

    cdef int _FAILURE_
    cdef int _FALSE_
    cdef int _TRUE_

    ctypedef char FileName[_FILENAMESIZE_]
    cdef struct matters:
        int has_cltp_nc
        int has_cltp_sh
        int cltp_size
        int cltp_grid_size
        int cltp_index_nc
        int cltp_index_sh
        int* window_size
        int** window_index_start
        int** window_index_end
        int ic_ic_size
        
        double* sampled_sources
        int size_fft_input
        int size_fft_cutoff
        double* k_sampling
        double* logk_sampling
        double deltalogk
        int uses_separability
        int matter_verbose
        int uses_intxi_logarithmic
        int tau_size
        double* growth_factor_tau
        double* tau_sampling
        int has_cls
        int num_windows
        int* num_windows_per_cltp
        int non_diag
        int tw_size
        int integrated_tw_size
        double* tw_sampling
        double* tw_weights
        double* integrated_tw_sampling
        double* integrated_tw_weights
        double* exp_integrated_tw_sampling
        double* tw_max
        double* tw_min
        double* fft_real
        double* fft_imag
        int l_lss_max
        double bias
        int ptw_size
        int ptw_integrated_size
        double** ptw_window
        double** ptw_integrated_window
        int radtp_size_total
        double* ptw_sampling
        double* ptw_integrated_sampling
        double l_logstep
        double l_linstep
        int has_unintegrated_windows
        int has_integrated_windows
        int uses_limber_approximation
        int t_size
        int t_spline_size
        double tau0
        int uses_bessel_store
        
        
        ErrorMsg error_message

    void matter_free(void*)
    int matter_init(void*)
    int matter_cl_at_l(void*, double*, int, double**)
