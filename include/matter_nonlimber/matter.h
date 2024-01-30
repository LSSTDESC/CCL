/** @file New way of calculating angular power spectra
 *
 * Nils Schöneberg, 18.10.2017 (edited for DESC until 20.01.2021)
 *
 */
#ifndef __MATTER__
#define __MATTER__
#include "fft.h"
#include "common.h"


#define _SELECTION_NUM_MAX_ 100
enum selection_type {gaussian,tophat,dirac};
enum matter_integration_method {matter_integrate_tw_t,matter_integrate_tw_logt};
enum matter_k_weight_method {matter_k_weights_gaussian,matter_k_weights_step};
struct matters{
  short matter_verbose; /**< flag regulating the amount of information sent to standard output (none if set to zero) */
  ErrorMsg error_message; /**< zone for writing error messages */

  /**
   * Use/Has/Allow flags, general
   * */
  short has_cls;
  short uses_separability;

  short uses_intxi_interpolation;
  short uses_intxi_symmetrized;
  short uses_intxi_logarithmic;
  short uses_intxi_asymptotic;

  short uses_rsd_combination;
  short uses_density_splitting;
  short has_integrated_windows;
  short has_unintegrated_windows;
  short has_window_differentiation;
  short uses_limber_approximation;
  short uses_relative_factors;
  short uses_all_l_sampling;
  short uses_bessel_analytic_integration;
  short uses_lensing_reduction;
  short uses_filon_clenshaw_curtis;

  short uses_bessel_storeall;
  short uses_bessel_store;

  int* is_non_zero;
  int non_diag;
  enum matter_integration_method uses_integration;
  /*
   * Tau sampling and tau, other general quantities
   * */
  double* tau_sampling; /**< The tau sampling of the sources */
  double tau0;

  int tau_size_max; //The limit to growing tau_size
  int tau_size; //The number of samples in tau space
  int tau_grid_size; // = tau_size*(tau_size+1)/2;
  int index_tau_perturbs_beginning; //Index of start of tau_matter sampling in tau_perturbs


  /**
   * ICs
   * */
  int ic_size;
  int ic_ic_size;

  /**
   * Type indices and type 'has' conditions
   * */

  int stp_index_delta_m;
  short has_stp_delta_m;
  int stp_index_theta_m;
  short has_doppler_terms;
  short has_redshift_space_distortion;
  short has_gravitational_terms;
  int stp_index_phi;
  int stp_index_phi_prime;
  short has_lensing_terms;
  short has_cl_shear;
  short has_stp_phi_plus_psi;
  int stp_index_phi_plus_psi;
  int stp_index_psi;

  int stp_size;
  int stp_grid_size;

  int radtp_dens;
  int radtp_dens1;
  int radtp_dens2;
  int radtp_nclens;
  int radtp_dop1;
  int radtp_dop2;
  int radtp_rsd;
  int radtp_rsd_combined;
  int radtp_g1;
  int radtp_g2;
  int radtp_g3;
  int radtp_g4;
  int radtp_g5;
  int radtp_combined;

  int radtp_shlens;

  int* radtp_of_bitp_size;
  int** radtps_of_bitp;
  int radtp_size_total;
  int radtp_grid_size;

  int bitp_index_normal;
  short has_bitp_normal;
  int bitp_index_lfactor;
  short has_bitp_lfactor;
  int bitp_index_nu_reduced;
  short has_bitp_nu_reduced;

  short has_tilt_normal;
  int tilt_index_normal;
  short has_tilt_reduced;
  int tilt_index_reduced;

  int bitp_size;
  int tilt_size;
  int tilt_grid_size;

  int* index_perturb_tp_of_stp;
  int* index_stp_of_radtp;

  /**
   * Radial types (bessel or bessel derivatives) and
   * Cl types
   * */
  int cltp_size;
  int cltp_grid_size;
  short has_cltp_nc;
  int cltp_index_nc;
  short has_cltp_sh;
  int cltp_index_sh;


  /**
   * Window max and mins
   * */
  double z_max;
  double tau_max;
  double tau_min; //Minimum tau given by normal window functions
  double* tw_max;
  double* tw_min;


  /**
   * FFT size
   * */
  int size_fft_input;
  int size_fft_result;
  int size_fft_cutoff;
  struct FFT_plan** FFT_plan;

  /**
   * Relative factors
   * */
  double* relative_factors;
  int index_relative_stp;

  /**
   * Growth factor
   * */
  double* growth_factor; //Sampled in tw
  double * growth_factor_tau; // Sampled in tau
  double * ddgrowth_factor_tau; //Sampled in tau
  double k_weight_k_max;
  double k_weight_k_min;
  enum matter_k_weight_method k_weight_mode;

  /**
   * k sampling, normal and extrapolated
   * */
  short allow_extrapolation;
  short extrapolation_type;

  double* k_sampling;
  double* logk_sampling; // Of size size_fft_coeff due to the nature of fft
  double logmink;
  double deltalogk;

  double k_max;
  double k_max_extr;

  int k_size;

  /**
   * Window sampling and windows
   * */

  double small_log_offset;
  int num_windows;           // Maximum number of windows over all Cl types
  int num_window_grid;
  int* num_windows_per_cltp; // Number of windows for each Cl type
  int tw_size; //Length in steps of tau-indices of the window
  double* tw_sampling; //[index_wd*tw_size+index_tw]The tau values sampled for the windows
  double* tw_weights; //Trapezoidal weights for tau window integration

  double* exp_integrated_tw_sampling; //exp(integrated_tw_sampling), ONLY in case of logarithmic sampling
  double* integrated_tw_sampling;
  double* integrated_tw_weights;
  int integrated_tw_size;

  int ptw_size;
  double* ptw_sampling;
  double* ptw_weights;
  double* ptw_orig_window;

  int ptw_integrated_size;
  double* ptw_integrated_sampling;
  double* ptw_integrated_weights;

  double** ptw_window;
  double** ptw_dwindow;
  double** ptw_ddwindow;

  /**
   * Intxi
   * */


  double** intxi_real;
  double** intxi_imag;

  double** intxi_spline_real;
  double** intxi_spline_imag;
  double** ddintxi_spline_real;
  double** ddintxi_spline_imag;

  /**
   * Short pvecback
   * */
  double* short_pvecback;//Used to obtain a(tau) to get z(tau)
  int selection;
  int has_nc_rsd;
  double* k_file;
  double k_min_file;
  int k_size_file;
  double* selection_mean;
  double* selection_width;
  double selection_cut_at_sigma;
  double selection_tophat_edge;
  int l_lss_max;
  double l_logstep;
  double l_linstep;
  double angular_rescaling;//=1
  double a_today;//=1
  int index_bg_a;
  int index_bg_H;
  int index_bg_H_prime;
  int bg_size_short;
  int has_source_delta_cb;
  int index_tp_delta_cb;
  int index_tp_delta_m;
  int index_tp_theta_m;
  int index_tp_phi;
  int index_tp_phi_prime;
  int index_tp_psi;
  int index_tp_phi_plus_psi;
  int has_ncdm;
  double *** sources;
  double ** nl_corr_density;
  int index_pk_m;
  int index_pk_cb;
  int* tp_size;
  int selection_num;
  
  
  
  /* ******************************************* */
  double* sampled_sources;
  double* fft_real;
  double* fft_imag;
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  /**
   * T sampling
   * */
  int t_size;
  double* t_sampling;
  double* t_weights;

  int t_spline_size;
  double* t_spline_sampling;

  /**
   * Hypergeometric/Bessel integrals coefficients
   * */
  double bias;
  double* nu_real;
  double* nu_imag;
  double bessel_imag_offset;

  /**
   * Hypergeometric/Bessel arrays
   * */
  int bi_wanted_samples;
  FileName bessel_file_name;
  double*** bi_real; //Bessel integral real
  double*** bi_imag; //Bessel integral imag
  double*** ddbi_real;
  double*** ddbi_imag;
  int** bi_size;
  double* bi_sampling;
  //[l*size_fft_coeff+ nu], [t]
  double** bi_max;
  double bi_maximal_t_offset;
  int bessel_recursion_t_size;

  /**
   * Experimental
   * */
  int bi_global_size;
  double* bi_global_sampling;
  double* chi_global_sampling;
  int chi_global_size;
  double*** cldd;
  double*** dl;
  /**
   * l sampling
   * */
  int l_size;
  int l_size_recursion;
  double* l_sampling;
  double* l_sampling_recursion;


  /**
   * Cl's
   * */
  double ** cl;
  double ** ddcl;
  int* window_size; //Number of (i,j) combinations for given cltp
  int** window_index_start;//Start of j for given i,cltp
  int** window_index_end;//Final value of j for given i,cltp

  /**
   * Selection/Window values, and other adjustments
   * */
  double selection_bias[_SELECTION_NUM_MAX_];               /**< light-to-mass bias in the transfer function of density number count */
  double selection_magnification_bias[_SELECTION_NUM_MAX_]; /**< magnification bias in the transfer function of density number count */
  short has_nz_file;     /**< Has dN/dz (selection function) input file? */
  short has_nz_analytic; /**< Use analytic form for dN/dz (selection function) distribution? */
  FileName nz_file_name; /**< dN/dz (selection function) input file name */
  int nz_size;           /**< number of redshift values in input tabulated selection function */
  double * nz_z;         /**< redshift values in input tabulated selection function */
  double * nz_nz;        /**< input tabulated values of selection function */
  double * nz_ddnz;      /**< second derivatives in splined selection function*/
  short has_nz_evo_file;      /**< Has dN/dz (evolution function) input file? */
  short has_nz_evo_analytic;  /**< Use analytic form for dN/dz (evolution function) distribution? */
  FileName nz_evo_file_name;  /**< dN/dz (evolution function) input file name */
  int nz_evo_size;            /**< number of redshift values in input tabulated evolution function */
  double * nz_evo_z;          /**< redshift values in input tabulated evolution function */
  double * nz_evo_nz;         /**< input tabulated values of evolution function */
  double * nz_evo_dlog_nz;    /**< log of tabulated values of evolution function */
  double * nz_evo_dd_dlog_nz; /**< second derivatives in splined log of evolution function */
};
struct matters_vector{

  double* integrand_real;
  double* integrand_imag;

  double** window_fft_real;
  double** window_fft_imag;
};
struct matters_workspace{

  double** intxi_spline_real;
  double** intxi_spline_imag;
  double** ddintxi_spline_real;
  double** ddintxi_spline_imag;

  double** intxi_real;
  double** intxi_imag;

  double* fft_coeff_real;
  double* fft_coeff_imag;

  double*** window_bessel_real;
  double*** window_bessel_imag;

  int index_ic1;
  int index_ic2;
  int index_ic1_ic2;

  int index_wd1;
  int index_wd2;
  int index_wd1_wd2;

  int window_counter; //Storage position in cl array

  int index_stp1;
  int index_stp2;
  int index_stp1_stp2;
  int index_stp2_stp1;

  int index_cltp1;
  int index_cltp2;
  int index_cltp1_cltp2;

  int index_radtp1;
  int index_radtp2;

  int index_tilt1_tilt2;

  double* tau_sampling; /**< The tau sampling of actual integration */
  double* tau_weights;
  int tau_size;

  double t_min;
  double t_max;

  double* pref_real;
  double* pref_imag;
  int tau_max_size;

  double** integrand_real;
  double** integrand_imag;

  int is_integrated_radtp1;
  int is_integrated_radtp2;

  int N_threads;
  struct matters_vector** pmv; /**< array of pointers, one for each thread */

  int window_offset;

};
#ifdef __cplusplus
extern "C" {
#endif
  int matter_init(
                  struct matters * pma
                  );
  int matter_free(
                  struct matters * pma
                  );
  int matter_obtain_l_sampling(
                  struct matters * pma
                  );
  int matter_obtain_indices(
                  struct matters* pma
                  );
  int matter_obtain_coeff_sampling(
                  struct matters * pma
                  );
  int matter_get_prepared_window_at(
                  struct matters* pma,
                  double tau,
                  int index_ic,
                  int index_radtp,
                  int index_wd,
                  int* last,
                  int derivative_type,
                  double* win_val
                  );
  int matter_obtain_bi_indices(
                  struct matters* pma
                  );
  int matter_obtain_nonseparability(
                  struct matters* pma,
                  double ** fft_coeff_real,
                  double ** fft_coeff_imag
                  );
  int matter_spline_cls(
                  struct matters* pma
                  );
  int matter_integrate_cl(
                  struct matters* pma,
                  double * fft_coeff_real,
                  double * fft_coeff_imag
                  );
  int matter_spline_bessel_integrals_recursion(
                  struct matters * pma
                  );
  int matter_interpolate_spline_growing_hunt(
                  double * x_array,
                  int n_lines,
                  double * array,
                  double * array_splined,
                  int n_columns,
                  double x,
                  int * last_index,
                  double * result,
                  ErrorMsg errmsg
                  );
  int matter_spline_prepare_hunt(
                  double* x_array,
                  int x_size,
                  double x,
                  int* last,
                  ErrorMsg errmsg
                  );
  int matter_cl_at_l(
                  struct matters* pma,
                  double * l_array,
                  int l_size,
                  double ** cl_tot
                  );
  int matter_get_bessel_limber(
                  struct matters* pma,
                  int index_l,
                  struct matters_workspace * pmw
                  );
  int matter_get_derivative_type(
                  struct matters* pma,
                  int* derivative_type1,
                  int* derivative_type2,
                  int index_radtp1,
                  int index_radtp2
                  );
  int matter_precompute_chit_factors(
                  struct matters* pma,
                  int index_wd,
                  double* pref_real,
                  double* pref_imag,
                  struct matters_workspace* pmw);
  int matter_asymptote(
                  struct matters* pma,
                  double t,
                  int index_wd1,
                  int index_wd2,
                  double* result
                  );
  int matter_spline_hunt(
                  double* x_array,
                  int x_size,
                  double x,
                  int* last,
                  double* h,
                  double* a,
                  double* b,
                  ErrorMsg errmsg
                  );
  int matter_obtain_t_sampling(
                  struct matters* pma
                  );
  int matter_integrate_each(
                  struct matters* pma,
                  struct matters_workspace * pmw
                  );
  int matter_get_bessel_fort_parallel(
                  struct matters* pma,
                  int index_l,
                  struct matters_workspace* pmw
                  );
  int matter_integrate_cosmo(
                  struct matters* pma,
                  struct matters_workspace* pmw
                  );
  int matter_obtain_bessel_recursion_parallel(
                  struct matters* pma
                  );
  int matter_get_bessel_fort_parallel_integrated(
                  struct matters* pma,
                  int index_l,
                  struct matters_workspace* pmw
                  );
  int matter_get_half_integrand(
                  struct matters* pma,
                  double t,
                  int index_ic1,
                  int index_ic2,
                  int index_radtp1,
                  int index_radtp2,
                  int index_stp1_stp2,
                  int index_wd1,
                  int index_wd2,
                  double* integrand_real,
                  double* integrand_imag,
                  double** wint_fft_real,
                  double** wint_fft_imag,
                  struct matters_workspace* pmw
                  );
  int matter_get_ttau_integrand(
                  struct matters* pma,
                  double t,
                  int index_ic1,
                  int index_ic2,
                  int index_radtp1,
                  int index_radtp2,
                  int index_stp1_stp2,
                  int index_wd1,
                  int index_wd2,
                  double* integrand_real,
                  double* integrand_imag,
                  double** wint_fft_real,
                  double** wint_fft_imag,
                  struct matters_workspace* pmw
                  );
  int matter_FFTlog_perturbation_sources_parallel(
                  struct matters * pma,
                  double ** fft_coeff_real,
                  double ** fft_coeff_imag
                  );
  int matter_swap_workspace(struct matters_workspace* pmw);
  int matter_vector_alloc(
                  struct matters* pma,
                  struct matters_workspace* pmw);
  int matter_vector_free(
                  struct matters* pma,
                  struct matters_workspace* pmw);
  int matter_read_bessel_integrals(struct matters* pma);
  int matter_write_bessel_integrals(struct matters* pma);
  int matter_read_bessel_file_correct(struct matters* pma,short* is_correct_file);
  int matter_obtain_window_indices(struct matters* pma);
#ifdef __cplusplus
}
#endif


#define CHUNK_SIZE 4
#define MATTER_REWRITE_PRINTING _FALSE_

#define MATTER_VERBOSITY_TIMING 1
#define MATTER_VERBOSITY_INDICES 2
#define MATTER_VERBOSITY_FUNCTIONS 3
#define MATTER_VERBOSITY_PARAMETERS 2
#define MATTER_VERBOSITY_RANGES 5
#define MATTER_VERBOSITY_BESSEL 5
#define MATTER_VERBOSITY_CLCALCULATION 4
#define MATTER_VERBOSITY_CLCALCULATION_PARTIAL 5
#define MATTER_VERBOSITY_CLRESULTS 4
#define MATTER_VERBOSITY_DELETE 5

#define matter_is_index(index_from,index_to,condition)              \
((condition) && ((index_from)==(index_to)))

#define matter_is_integrated(index_radtp)                           \
((pma->has_lensing_terms && index_radtp == pma->radtp_nclens)       \
|| (pma->has_gravitational_terms &&                                 \
(index_radtp == pma->radtp_g4 || index_radtp == pma->radtp_g5) )    \
|| (pma->has_cl_shear && index_radtp == pma->radtp_shlens)          \
)

#define matter_get_t(index_t)                                       \
if(integrate_logarithmically && !pma->uses_limber_approximation){   \
  double y = pma->t_spline_sampling[(index_t)]*(y_max-y_min)+y_min; \
  t = 1.0-exp(-y);                                                  \
}else if(!pma->uses_limber_approximation){                          \
  t = pma->t_spline_sampling[(index_t)]*(t_max-t_min)+t_min;        \
}                                                                   \
else{                                                               \
  t = 1.0;                                                          \
}
#define matter_get_t_orig(index_t)                                  \
if(integrate_logarithmically){                                      \
  double y = pma->t_sampling[index_t]*(y_max-y_min)+y_min;          \
  t = 1.0-exp(-y);                                                  \
}else{                                                              \
  t = pma->t_sampling[index_t]*(t_max-t_min)+t_min;                 \
}
#define matter_get_tij_limits(index_wd1,index_wd2)                    \
t_min = MAX((pma->tau0-pma->tw_max[index_wd1])/(pma->tau0-pma->tw_min[index_wd2]),0.0+pma->bi_maximal_t_offset);\
t_max = MIN((pma->tau0-pma->tw_min[index_wd1])/(pma->tau0-pma->tw_max[index_wd2]),1.0-pma->bi_maximal_t_offset);
#define matter_get_t_limits(index_wd1,index_wd2)                    \
matter_get_tij_limits(index_wd2,index_wd1)                          \
if(t_min>t_max){matter_get_tij_limits(index_wd1,index_wd2);}        \
else{                                                               \
  double temp_t_min = t_min; double temp_t_max=t_max;               \
  matter_get_tij_limits(index_wd1,index_wd2);                       \
  t_min = MIN(temp_t_min,t_min); t_max = MAX(temp_t_max,t_max);     \
}
/* macro for re-allocating memory, returning error if it failed */
#define class_realloc_parallel(pointer, newname, size, error_message_output)  {                                  \
  if(abort==_FALSE_){                                                                                            \
    pointer=realloc(newname,size);                                                                               \
    if (pointer == NULL) {                                                                                       \
      int size_int;                                                                                              \
      size_int = size;                                                                                           \
      class_alloc_message(error_message_output,#pointer, size_int);                                              \
      abort=_TRUE_;                                                                                              \
    }                                                                                                            \
  }                                                                                                              \
}
#endif
