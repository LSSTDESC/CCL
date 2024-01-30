/** @file New way of calculating angular power spectra
 *
 * Nils Schöneberg, 16.10.2017 (edited for DESC until 20.01.2021)
 *
 */

#include "matter.h"
#include "hypergeom.h"
#include "arrays.h"
#include "fft.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/**
 * Anisotropy matter power spectra \f$ C_l\f$'s for all types, windows and initial conditions.
 * The mode is always scalar.
 *
 * This routine evaluates all the \f$C_l\f$'s at a given value of l by
 * interpolating in the pre-computed table. When relevant, it also
 * sums over all initial conditions.
 *
 * This function can be called from whatever module at whatever time, provided that
 * matter_init() has been called before, and matter_free() has not
 * been called yet.
 *
 * @param pma        Input: pointer to matter structure (containing pre-computed table)
 * @param l          Input: multipole array
 * @param l_size     Input: length of multipole array
 * @param cl_tot     Output: array with argument cl_tot[index_cltp_grid][index_l*pma->window_size[index_cltp_grid]+index_wd_grid] (must be already allocated)
 * @return the error status
 */
int matter_cl_at_l(
                  struct matters* pma,
                  double * l_array,
                  int l_size,
                  double ** cl_tot
                  ) {
  /**
   * Initialize local variables
   * */

  int last_index;
  int index_cltp_grid,index_l;
  int index_wd_grid;
  int offset = 0;
  last_index = 0;


  /** Test for having calculated the cls  */
  class_test(pma->has_cls && pma->l_size<=0,pma->error_message,"Matter was never calculated. Cannot obtain Cl's");



  /**
   * We set those Cl's above l_max to 0
   * */
  for (index_cltp_grid=0; index_cltp_grid<pma->cltp_grid_size; index_cltp_grid++){
    for (index_l=0;index_l<l_size;++index_l){
      class_test(l_array[index_l]>pma->l_sampling[pma->l_size-1], pma->error_message, "Requested l=%g is larger than maximum l in matter : %g \n",l_array[index_l],pma->l_sampling[pma->l_size-1]);
      /**
       * Interpolate at given value of l for all windows at once
       * */
      class_call(matter_interpolate_spline_growing_hunt(
                                          pma->l_sampling,
                                          pma->l_size,
                                          pma->cl[index_cltp_grid],
                                          pma->ddcl[index_cltp_grid],
                                          pma->window_size[index_cltp_grid],
                                          l_array[index_l],
                                          &last_index,
                                          cl_tot[index_cltp_grid]+index_l*pma->window_size[index_cltp_grid],
                                          pma->error_message),
                 pma->error_message,
                 pma->error_message);
    }
  }
  return _SUCCESS_;
}





/**
 * Initialize the matter construct.
 *
 * This initializes the matter construct, and computes the \f$C_l\f$'s
 * of flat space for number count/shear.
 *
 * This function must be called before any internal calls, and before
 * matter_free()
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to thermodynamics structure
 * @param ppt        Input: pointer to perturbation structure
 * @param ppm        Input: pointer to primordial structure
 * @param pnl        Input: pointer to nonlinear structure
 * @param pma        Input/Output: pointer to matters structure
 * @return the error status
 */
int matter_init(
                struct matters * pma
             ){

  /** Summary: */

  /** - Check for computational flags */
  if (pma->has_cls == _FALSE_){
    /* In this case, the user requested deactivation of this module, skip everything */
    return _SUCCESS_;
  }
  /* Otherwise start computation */
  else if (pma->matter_verbose > 0){
    fprintf(stdout,"Computing matter spectra\n");
  }
  if(pma->matter_verbose > MATTER_VERBOSITY_PARAMETERS){
    printf(" -> Verbosity set to %i \n",pma->matter_verbose);
  }


#ifdef _OPENMP
  double point0_time = omp_get_wtime();
#endif



  /**
   *  - Copy parameters from the other structs
   * */
  /* (These should be assigned BEFORE any call to matter_obtain_indices) */
  pma->tau_size_max = 100;
  pma->tau_size = MIN(2*(int)(0.5*pma->tau_size+0.25),pma->tau_size_max);
  pma->angular_rescaling = 1.;
  /**
   *  - Set "fixed" flags
   * */

  pma->uses_density_splitting = _FALSE_;
  pma->uses_all_l_sampling = _FALSE_;
  pma->uses_lensing_reduction = _TRUE_;
  pma->uses_rsd_combination = _TRUE_;
  pma->uses_limber_approximation = _FALSE_;
  pma->uses_relative_factors = _FALSE_;

  pma->uses_bessel_storeall = _FALSE_;

  pma->uses_integration = matter_integrate_tw_t;

  pma->uses_intxi_symmetrized = _TRUE_;
  pma->uses_intxi_asymptotic = _FALSE_;
  pma->uses_intxi_logarithmic = _TRUE_;
  pma->uses_intxi_interpolation = _TRUE_;

  /**
   *  - Obtain indices required for later evaluation
   * */
  class_call(matter_obtain_indices(pma),
             pma->error_message,
             pma->error_message);
  class_call(matter_obtain_window_indices(pma),
             pma->error_message,
             pma->error_message);
  class_call(matter_obtain_bi_indices(pma),
             pma->error_message,
             pma->error_message);


  //Tw size: 30 for lens and dens
  //IntTw size: 75 for lens and dens
  //t spline size: 50 for lens, 20 for dens
  pma->bessel_recursion_t_size = 200;

  pma->size_fft_result = pma->size_fft_cutoff;
  pma->bi_maximal_t_offset = 1.e-7;
  
  pma->tau_grid_size = pma->tau_size*pma->tau_size;

  //Offset for keeping numerical instabilities for
  //log(x) small for x->0 (e.g. exp(log(x1)-log(x2) ) = 0 for x1 approx = 0, not NaN
  //Also used for setting exp(log(x))!=x to a value that does not over/underflow the bounds
  pma->small_log_offset = 1e-10;

  pma->has_nc_rsd = _FALSE_;
  /**
   *  - Test wether the defined combinations would give a valid caclulation
   * */
  pma->selection = gaussian;
  class_test(pma->selection == dirac && pma->has_nc_rsd,
             pma->error_message,
             "Including redshift space distortions for dirac functions not yet implemented.");
  class_test(pma->selection == tophat && pma->has_nc_rsd,
             pma->error_message,
            "Including redshift space distortions for tophat functions not yet implemented.");
  class_test(pma->size_fft_cutoff>pma->size_fft_result,
             pma->error_message,
             "the coefficient cutoff size (%i) has to be smaller or equal to the result size (%i)",
             pma->size_fft_cutoff,pma->size_fft_result);
  class_test(pma->tau_size%2!=0,
             pma->error_message,
             "The tau_size parameter currently has to be a multiple of 2");
  /* Done testing parameter combinations */


#ifdef _OPENMP
  double point1_time = omp_get_wtime();
#endif


#ifdef _OPENMP
  double point2_time = omp_get_wtime();
#endif


#ifdef _OPENMP
  double point3_time = omp_get_wtime();
#endif





  /**
   *  - Now all sources are obtained, and we can proceed
   *    with calculating the FFT in logarithmic k space
   * */
  double* fft_coeff_real;
  double* fft_coeff_imag;
  class_call(matter_obtain_coeff_sampling(pma),
             pma->error_message,
             pma->error_message);
  class_call(matter_FFTlog_perturbation_sources_parallel(pma,&fft_coeff_real,&fft_coeff_imag),
             pma->error_message,
             pma->error_message);

  if(!pma->uses_separability){
    /*
     * This function !replaces! the fft coefficients with their (nearly) constant counterparts
     * The originals can be obtained by re-multiplying with the growth factors
     * */
    class_call(matter_obtain_nonseparability(
                                          pma,
                                          &fft_coeff_real,
                                          &fft_coeff_imag),
               pma->error_message,
               pma->error_message);

  }


  pma->fft_real = fft_coeff_real;
  pma->fft_imag = fft_coeff_imag;


#ifdef _OPENMP
  double point4_time = omp_get_wtime();
#endif



  class_call(matter_obtain_l_sampling(pma),
             pma->error_message,
             pma->error_message);
  /**
   *  - There are two big ways of obtaining the bessel integrals
   *
   *    1) Use the recursion relation of the bessel integrals
   *       This proves to be really fast, and surprisingly even more accurate
   *
   *    2) Using the direct representations through taylor series
   *       This older method proves to become unreliable due to
   *       floating point arithmetics, especially around
   *       high imaginary parts in nu, large l, and t of around 0.9-0.99
   *
   *    We first obtain the integrals for a pre-made grid of t values,
   *    after which we spline them for exactly those t values that we require evaluation at
   *
   *    This way the initial t sampling is independent of l and nu,
   *    while the final one can and does depend on the precise nature
   *    of the window functions etc.
   * */
  short is_correct_file = pma->uses_bessel_store;
  class_call(matter_read_bessel_file_correct(pma,&is_correct_file),
             pma->error_message,
             pma->error_message);
  if(!pma->uses_limber_approximation && !is_correct_file){
    /**
     *  - Obtain the bessel integrals
     * */
    class_call(matter_obtain_bessel_recursion_parallel(pma),
               pma->error_message,
               pma->error_message);
    /**
     *  - Spline bessel integrals
     * */
    class_call(matter_spline_bessel_integrals_recursion(pma),
             pma->error_message,
             pma->error_message);

    if(pma->uses_bessel_store){
      class_call(matter_write_bessel_integrals(pma),pma->error_message,pma->error_message);
    }
  }
  if(is_correct_file){
    class_call(matter_read_bessel_integrals(pma),pma->error_message,pma->error_message);
  }
  //Ifend obtain bessel integrals
  /* Done getting Bessel integrals and l sampling */




#ifdef _OPENMP
  double point5_time = omp_get_wtime();
#endif




#ifdef _OPENMP
  double point6_time = omp_get_wtime();
#endif





  /* Get t sampling */
  class_call(matter_obtain_t_sampling(pma),
             pma->error_message,
             pma->error_message);



#ifdef _OPENMP
  double point7_time = omp_get_wtime();
#endif








  /**
   *  - Now we have truly assembled all ingredients
   *    to integrate the final \f$C_\ell\f$'s
   *    a) We have the window functions (including growth factors)
   *    b) We have the power law exponents
   *    c) We have the FFT coefficients
   *    d) We have the bessel integrals
   *    e) We have all relevant sampling grids
   *
   *    Thus we are finally able to obtain the \f$C_\ell\f$'s
   * */
  class_call(matter_integrate_cl(pma,
                                 fft_coeff_real,
                                 fft_coeff_imag),
             pma->error_message,
             pma->error_message);
  /* Done integrating Cl's */




#ifdef _OPENMP
  double point8_time = omp_get_wtime();
#endif


  /**
   *  - Finally we spline the \f$C_\ell\f$'s to interpolate
   *    for all \f$\ell\f$'s
   * */
  class_call(matter_spline_cls(pma),
             pma->error_message,
             pma->error_message);
  /* Done splining */


#ifdef _OPENMP
  double point9_time = omp_get_wtime();
#endif





  /**
   *  - If desired, we give a summary of the program's running configuration
   *    and/or the timings of each part of the program
   * */
  if(pma->matter_verbose > MATTER_VERBOSITY_PARAMETERS){
    printf("\n\n\n PARAMETER SUMMARY \n\n ");
    printf("Calculationary parameters \n");
    printf(" -> tilt == %.10e \n",pma->bias);
    printf(" -> nu_imag max == %.10e \n",pma->nu_imag[pma->size_fft_result-1]);
    printf(" -> nu_imag step == %.10e \n",pma->nu_imag[1]);
    printf(" -> k_max == %.10e \n",pma->k_sampling[pma->size_fft_input-1]);
    printf(" -> k_min == %.10e \n",pma->k_sampling[0]);
    printf(" -> delta log(k) == %.10e \n \n\n",pma->deltalogk);
    printf(" -> tau0 == %.10e \n",pma->tau0);

    printf("Parameter counts \n");
    printf(" -> Number of types %i \n",pma->stp_size);
    printf(" -> Number of radials %i \n",pma->radtp_size_total);
    printf(" -> Number of bessel integrals %i \n",pma->bitp_size);
    printf(" -> Number of tilts %i \n",pma->tilt_size);

    printf("Parameter options as follows : \n");
    printf(" -> Parameter '%s' has value %s \n","has_cls",(pma->has_cls?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses integration",(pma->uses_integration==matter_integrate_tw_t?"tw_t":"tw_logt"));
    printf(" -> Parameter '%s' has value %s \n","uses seperability",(pma->uses_separability?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","allow extrapolation",(pma->allow_extrapolation?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses density spltting",(pma->uses_density_splitting?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses intxi_interpolation",(pma->uses_intxi_interpolation?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses intxi_logarithmic",(pma->uses_intxi_logarithmic?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses intxi_symmetric",(pma->uses_intxi_symmetrized?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses intxi_asymptotic",(pma->uses_intxi_asymptotic?"TRUE":"FALSE"));
    //printf(" -> Parameter '%s' has value %s \n","uses analytic bessel",(pma->uses_bessel_analytic_integration?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses all ell",(pma->uses_all_l_sampling?"TRUE":"FALSE"));

    printf(" -> Parameter '%s' has value %s \n","uses RSD combination",(pma->uses_rsd_combination?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses relative factors",(pma->uses_relative_factors?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %s \n","uses limber approximation",(pma->uses_limber_approximation?"TRUE":"FALSE"));
    printf(" -> Parameter '%s' has value %i \n","window number",pma->num_windows);
    printf(" -> Parameter '%s' has value %i \n","nondiagonals",pma->non_diag);

    printf(" -> Parameter '%s' has value %i \n","tw size",pma->tw_size);
    printf(" -> Parameter '%s' has value %i \n","tw integrated size",pma->integrated_tw_size);
    printf(" -> Parameter '%s' has value %i \n","t size",pma->t_size);
    printf(" -> Parameter '%s' has value %i \n","max coeff",pma->size_fft_cutoff);

  }

#ifdef _OPENMP
  if(pma->matter_verbose > MATTER_VERBOSITY_TIMING){
    printf("\n\n\n TIMING SUMMARY (for %10d spectra, %5d l values) \n\n",((pma->non_diag+1)*(2*pma->num_windows-pma->non_diag))/2,pma->l_size);
    printf("Initialization                   %15f seconds \n",point1_time-point0_time);
    printf("k,tau Sampling + Sources         %15f seconds \n",point2_time-point1_time);
    printf("Growth Factor + Sampled Sources  %15f seconds \n",point3_time-point2_time);
    printf("FFTlog of Sampled Sources        %15f seconds \n",point4_time-point3_time);
    printf("Bessel Integrals                 %15f seconds \n",point5_time-point4_time);
    printf("Calculating Window Functions     %15f seconds \n",point6_time-point5_time);
    printf("Resample Growth Factor           %15f seconds \n",point7_time-point6_time);
    printf("Integrating Cl's                 %15f seconds (%15f sec per spectrum)\n",point8_time-point7_time,(point8_time-point7_time)/(((pma->non_diag+1)*(2*pma->num_windows-pma->non_diag))/2));
    printf("Freeing resources + Spline Cl's  %15f seconds \n",point9_time-point8_time);
  }
#endif
  /* Done timing/parameter outputs */

#ifdef _OPENMP
  double point10_time = omp_get_wtime();
#endif



#ifdef _OPENMP
  if(pma->matter_verbose>MATTER_VERBOSITY_TIMING){
    printf("Matter took                      %15f seconds \n",point10_time-point0_time);
  }
  /* Done matter module */
#endif

  return _SUCCESS_;
}










/**
 * Free all memory occupied by the matter module
 *
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_free(
                struct matters * pma
              ) {
  int i,j;
  if(pma->has_cls){
    if(pma->matter_verbose>MATTER_VERBOSITY_FUNCTIONS){
      printf("Method :: Free \n");
    }
    if(pma->matter_verbose>MATTER_VERBOSITY_DELETE){
      printf("Freeing fft-related quantities \n");
    }
    //free(pma->logk_sampling);
    //free(pma->k_sampling);
    //free(pma->tau_sampling);

    if(pma->matter_verbose>MATTER_VERBOSITY_DELETE){
      printf("Freeing up window-related quantities \n");
    }
    //free(pma->tw_sampling);
    //free(pma->tw_weights);
    //free(pma->integrated_tw_sampling);
    //free(pma->integrated_tw_weights);
    //free(pma->exp_integrated_tw_sampling);

    //free(pma->tw_max);
    //free(pma->tw_min);

    free(pma->t_sampling);
    free(pma->t_weights);

    if(pma->uses_intxi_interpolation){
      free(pma->t_spline_sampling);
    }
    free(pma->fft_real);
    free(pma->fft_imag);


    if(pma->matter_verbose>MATTER_VERBOSITY_DELETE){
      printf("Freeing up bessel memory \n");
    }
    free(pma->nu_real);
    free(pma->nu_imag);
    if(!pma->uses_limber_approximation){
      for(j=0;j<pma->tilt_grid_size;++j){
        for(i=0;i<pma->l_size_recursion*pma->size_fft_cutoff;++i){
          int delta = (i/pma->size_fft_cutoff);
          int index = delta*pma->size_fft_result+(i-pma->size_fft_cutoff*delta);
          free(pma->bi_real[j][index]);
          free(pma->bi_imag[j][index]);
          free(pma->ddbi_real[j][index]);
          free(pma->ddbi_imag[j][index]);
        }
        free(pma->bi_real[j]);
        free(pma->bi_imag[j]);
        free(pma->ddbi_real[j]);
        free(pma->ddbi_imag[j]);
        free(pma->bi_size[j]);
        free(pma->bi_max[j]);
      }
      free(pma->bi_real);
      free(pma->bi_imag);
      free(pma->ddbi_real);
      free(pma->ddbi_imag);
      free(pma->bi_sampling);
      free(pma->bi_size);
      free(pma->bi_max);
    }

    if(pma->matter_verbose>MATTER_VERBOSITY_DELETE){
      printf("Freeing up general memory \n");
    }
    free(pma->l_sampling);

    if(pma->has_cltp_nc){
      if(pma->has_bitp_normal){
        free(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_normal]);
      }
      if(pma->has_bitp_nu_reduced){
        free(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced]);
      }
      if(pma->has_bitp_lfactor){
        free(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_lfactor]);
      }
    }
    if(pma->has_cltp_sh){
      if(pma->has_bitp_lfactor){
        free(pma->radtps_of_bitp[pma->cltp_index_sh*pma->bitp_size+pma->bitp_index_lfactor]);
      }
    }
    free(pma->radtps_of_bitp);
    free(pma->radtp_of_bitp_size);

    for(i=0;i<pma->cltp_grid_size;++i){
      free(pma->window_index_start[i]);
      free(pma->window_index_end[i]);
    }
    free(pma->window_index_start);
    free(pma->window_index_end);
    free(pma->window_size);

    for(i=0;i<pma->ic_ic_size;++i){
      for(j=0;j<pma->cltp_grid_size;++j){
        free(pma->cl[i*pma->cltp_grid_size+j]);
        free(pma->ddcl[i*pma->cltp_grid_size+j]);
      }
    }
    free(pma->cl);
    free(pma->ddcl);
  }
  return _SUCCESS_;
}

/**
 * Free workspace
 *
 * @param pma        Input: pointer to matter structure
 * @param pmw        Input: pointer to matter workspace structure
 * @return the error status
 */
int matter_workspace_free(struct matters* pma,
                          struct matters_workspace* pmw){
  int index_radtp1_radtp2,index_l,index_tilt_grid,index_coeff;

  free(pmw->pref_real);
  free(pmw->pref_imag);

  /**
   * Finally delete the intxi storage arrays again
   * */
  if(pma->uses_intxi_interpolation){
    for(index_radtp1_radtp2=0;index_radtp1_radtp2<pma->radtp_grid_size;++index_radtp1_radtp2){
      free(pmw->intxi_spline_real[index_radtp1_radtp2]);
      free(pmw->intxi_spline_imag[index_radtp1_radtp2]);
      free(pmw->ddintxi_spline_real[index_radtp1_radtp2]);
      free(pmw->ddintxi_spline_imag[index_radtp1_radtp2]);
    }
    free(pmw->intxi_spline_real);
    free(pmw->intxi_spline_imag);
    free(pmw->ddintxi_spline_real);
    free(pmw->ddintxi_spline_imag);
  }
  for(index_radtp1_radtp2=0;index_radtp1_radtp2<pma->radtp_grid_size;++index_radtp1_radtp2){
    free(pmw->intxi_real[index_radtp1_radtp2]);
    free(pmw->intxi_imag[index_radtp1_radtp2]);
  }
  free(pmw->intxi_real);
  free(pmw->intxi_imag);


  //End n_thread
  for(index_l=0;index_l<pma->l_size;++index_l){
    for(index_tilt_grid=0;index_tilt_grid<pma->tilt_grid_size;++index_tilt_grid){
      for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
        free(pmw->window_bessel_real[index_l][index_tilt_grid*pma->size_fft_result+index_coeff]);
        free(pmw->window_bessel_imag[index_l][index_tilt_grid*pma->size_fft_result+index_coeff]);
      }
      //End coeff
    }
    //End tilt grid
    free(pmw->window_bessel_real[index_l]);
    free(pmw->window_bessel_imag[index_l]);
  }
  //End l
  free(pmw->window_bessel_real);
  free(pmw->window_bessel_imag);


  return _SUCCESS_;
}

/**
 * Allocate workspace
 *
 * @param pma        Input: pointer to matter structure
 * @param pmw        Input: pointer to matter workspace structure
 * @return the error status
 */
int matter_workspace_alloc(struct matters* pma,
                           struct matters_workspace* pmw){
  int index_radtp1_radtp2,index_l,index_tilt_grid,index_coeff;

  /**
   * Prefactor allocations
   * */
  int tw_max_size = 0;
  if(pma->has_unintegrated_windows){
    tw_max_size = MAX(tw_max_size,pma->tw_size);
  }
  if(pma->has_integrated_windows){
    tw_max_size = MAX(tw_max_size,pma->integrated_tw_size);
  }
  pmw->tau_max_size = tw_max_size;


  if(pma->uses_intxi_symmetrized){
    class_alloc(pmw->pref_real,
                2*tw_max_size*pma->size_fft_result*sizeof(double),
                pma->error_message);
    class_alloc(pmw->pref_imag,
                2*tw_max_size*pma->size_fft_result*sizeof(double),
                pma->error_message);
  }
  else{
    class_alloc(pmw->pref_real,
                tw_max_size*pma->size_fft_result*sizeof(double),
                pma->error_message);
    class_alloc(pmw->pref_imag,
                tw_max_size*pma->size_fft_result*sizeof(double),
                pma->error_message);
  }


  /**
   * Now allocate local arrays to store the function f_n^{ij}(t) in
   *
   * These can theoretically become quite big,
   *   so we allocate a single one for every window and ic combination
   * */
  class_alloc(pmw->intxi_real,
              pma->radtp_grid_size*sizeof(double*),
              pma->error_message);
  class_alloc(pmw->intxi_imag,
              pma->radtp_grid_size*sizeof(double*),
              pma->error_message);
  for(index_radtp1_radtp2=0;index_radtp1_radtp2<pma->radtp_grid_size;++index_radtp1_radtp2){
    class_alloc(pmw->intxi_real[index_radtp1_radtp2],
                pma->size_fft_result*pma->t_size*sizeof(double),
                pma->error_message);
    class_alloc(pmw->intxi_imag[index_radtp1_radtp2],
                pma->size_fft_result*pma->t_size*sizeof(double),
                pma->error_message);
  }


  /**
   * If we desire interpolation, those arrays also have to be allocated
   * */
  if(pma->uses_intxi_interpolation){
    class_alloc(pmw->intxi_spline_real,
                pma->radtp_grid_size*sizeof(double*),
                pma->error_message);
    class_alloc(pmw->intxi_spline_imag,
                pma->radtp_grid_size*sizeof(double*),
                pma->error_message);
    class_alloc(pmw->ddintxi_spline_real,
                pma->radtp_grid_size*sizeof(double*),
                pma->error_message);
    class_alloc(pmw->ddintxi_spline_imag,
                pma->radtp_grid_size*sizeof(double*),
                pma->error_message);
    for(index_radtp1_radtp2=0;index_radtp1_radtp2<pma->radtp_grid_size;++index_radtp1_radtp2){
      class_alloc(pmw->intxi_spline_real[index_radtp1_radtp2],
                  pma->size_fft_result*pma->t_spline_size*sizeof(double*),
                  pma->error_message);
      class_alloc(pmw->intxi_spline_imag[index_radtp1_radtp2],
                  pma->size_fft_result*pma->t_spline_size*sizeof(double*),
                  pma->error_message);
      class_alloc(pmw->ddintxi_spline_real[index_radtp1_radtp2],
                  pma->size_fft_result*pma->t_spline_size*sizeof(double*),
                  pma->error_message);
      class_alloc(pmw->ddintxi_spline_imag[index_radtp1_radtp2],
                  pma->size_fft_result*pma->t_spline_size*sizeof(double*),
                  pma->error_message);
    }
  }

  /**
   * Now allocate bessel arrays
   * */
  class_alloc(pmw->window_bessel_real,
              pma->l_size*sizeof(double**),
              pma->error_message);
  class_alloc(pmw->window_bessel_imag,
              pma->l_size*sizeof(double**),
              pma->error_message);
  for(index_l=0;index_l<pma->l_size;++index_l){
    class_alloc(pmw->window_bessel_real[index_l],
                pma->tilt_grid_size*pma->size_fft_result*sizeof(double*),
                pma->error_message);
    class_alloc(pmw->window_bessel_imag[index_l],
                pma->tilt_grid_size*pma->size_fft_result*sizeof(double*),
                pma->error_message);
    for(index_tilt_grid=0;index_tilt_grid<pma->tilt_grid_size;++index_tilt_grid){
      for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
        class_alloc(pmw->window_bessel_real[index_l][index_tilt_grid*pma->size_fft_result+index_coeff],
                    (pma->has_integrated_windows?2:1)*pma->t_size*sizeof(double),
                    pma->error_message);
        class_alloc(pmw->window_bessel_imag[index_l][index_tilt_grid*pma->size_fft_result+index_coeff],
                    (pma->has_integrated_windows?2:1)*pma->t_size*sizeof(double),
                    pma->error_message);
      }
      //End coeff
    }
    //End tilt grid
  }
  //End l
  return _SUCCESS_;
}


/**
 * Allocate the matter vector within the workspace
 *
 * @param pma        Input: pointer to matter structure
 * @param pmw        Input: pointer to matter workspace structure
 * @return the error status
 */
int matter_vector_alloc(struct matters* pma,
                        struct matters_workspace* pmw){
  int n_thread,index_tw;

#ifdef _OPENMP
  pmw->N_threads = omp_get_max_threads();
#else
  pmw->N_threads = 1;
#endif

  class_alloc(pmw->pmv,
              pmw->N_threads*sizeof(struct matters_vector*),
              pma->error_message);
  for(n_thread=0;n_thread<pmw->N_threads;++n_thread){
    class_alloc(pmw->pmv[n_thread],
                sizeof(struct matters_vector),
                pma->error_message);
  }

  /**
   * First allocalte fft coefficient arrays
   * */
  for(n_thread=0;n_thread<pmw->N_threads;++n_thread){
    class_alloc(pmw->pmv[n_thread]->window_fft_real,
                2*pmw->tau_max_size*sizeof(double*),
                pma->error_message);
    class_alloc(pmw->pmv[n_thread]->window_fft_imag,
                2*pmw->tau_max_size*sizeof(double*),
                pma->error_message);
    if(!pma->uses_separability){
      for(index_tw=0;index_tw<2*pmw->tau_max_size;++index_tw){
          class_alloc(pmw->pmv[n_thread]->window_fft_real[index_tw],
                  pma->size_fft_result*sizeof(double),
                  pma->error_message);
          class_alloc(pmw->pmv[n_thread]->window_fft_imag[index_tw],
                  pma->size_fft_result*sizeof(double),
                  pma->error_message);
      }
      //End index_tw
    }
    //Ifend separability
    if(pma->uses_intxi_symmetrized){
      class_alloc(pmw->pmv[n_thread]->integrand_real,
                  2*pmw->tau_max_size*pma->size_fft_result*sizeof(double),
                  pma->error_message);
      class_alloc(pmw->pmv[n_thread]->integrand_imag,
                  2*pmw->tau_max_size*pma->size_fft_result*sizeof(double),
                  pma->error_message);
    }
    else{
      class_alloc(pmw->pmv[n_thread]->integrand_real,
                  pmw->tau_max_size*pma->size_fft_result*sizeof(double),
                  pma->error_message);
      class_alloc(pmw->pmv[n_thread]->integrand_imag,
                  pmw->tau_max_size*pma->size_fft_result*sizeof(double),
                  pma->error_message);
    }
  }
  return _SUCCESS_;
}


/**
 * Free matter vector within the matter workspace
 *
 * @param pma        Input: pointer to matter structure
 * @param pmw        Input: pointer to matter workspace structure
 * @return the error status
 */
int matter_vector_free(struct matters* pma,
                       struct matters_workspace* pmw){
  int n_thread,index_tw;
  for(n_thread=0;n_thread<pmw->N_threads;++n_thread){
    free(pmw->pmv[n_thread]->integrand_real);
    free(pmw->pmv[n_thread]->integrand_imag);
    if(!pma->uses_separability){
      for(index_tw=0;index_tw<2*pmw->tau_max_size;++index_tw){
        free(pmw->pmv[n_thread]->window_fft_real[index_tw]);
        free(pmw->pmv[n_thread]->window_fft_imag[index_tw]);
      }
      //End tw
    }
    //Ifend sep
    free(pmw->pmv[n_thread]->window_fft_real);
    free(pmw->pmv[n_thread]->window_fft_imag);
  }

  for(n_thread=0;n_thread<pmw->N_threads;++n_thread){
    free(pmw->pmv[n_thread]);
  }
  free(pmw->pmv);
  return _SUCCESS_;
}





/**
 * Spline the final Cl's
 *
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_spline_cls(
                      struct matters* pma
                      ){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Splining the final Cl's \n");
  }
  int index_cltp_grid;
  int index_wd1_wd2;
  class_alloc(pma->ddcl,
              pma->cltp_grid_size*sizeof(double*),
              pma->error_message);

  for(index_cltp_grid=0;index_cltp_grid<pma->cltp_grid_size;++index_cltp_grid){
    {
      class_alloc(pma->ddcl[index_cltp_grid],
                  pma->window_size[index_cltp_grid]*pma->l_size*sizeof(double),
                  pma->error_message);
      for(index_wd1_wd2=0;index_wd1_wd2<pma->window_size[index_cltp_grid];++index_wd1_wd2){
        array_spline_table_columns(pma->l_sampling,
                                   pma->l_size,
                                   pma->cl[index_cltp_grid]+index_wd1_wd2*pma->l_size,
                                   1,
                                   pma->ddcl[index_cltp_grid]+index_wd1_wd2*pma->l_size,
                                   _SPLINE_EST_DERIV_,
                                   pma->error_message);
      }
      //End wd
    }
    //End ic_ic
  }
  //End cl tp
  return _SUCCESS_;
}







/**
 * Obtain sampling for the fft coefficients
 *
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_obtain_coeff_sampling(
                               struct matters * pma
                              ){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Obtaining FFT coefficient sampling\n");
  }
  int index_coeff;
  int index_tilt1,index_tilt2,index_tilt1_tilt2;
  double current_tilt_offset;
  class_alloc(pma->nu_real,
              pma->tilt_grid_size*sizeof(double),
              pma->error_message);
  /**
   * The real part of the coefficient
   *  depends only on the tilt
   * We have to iterate thus through every
   *  possible combination of tilts.
   * */
  for(index_tilt1=0;index_tilt1<pma->tilt_size;++index_tilt1){
    for(index_tilt2=index_tilt1;index_tilt2<pma->tilt_size;++index_tilt2){
      index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);
      if(matter_is_index(index_tilt1,pma->tilt_index_normal,pma->has_tilt_normal)){
        if(matter_is_index(index_tilt2,pma->tilt_index_normal,pma->has_tilt_normal)){
          current_tilt_offset = 0.0;
        }
        else if(matter_is_index(index_tilt2,pma->tilt_index_reduced,pma->has_tilt_reduced)){
          current_tilt_offset = 2.0;
        }
        else{
          class_stop(pma->error_message,"Tilt index %i not recognized",index_tilt2);
        }
      }
      else if(matter_is_index(index_tilt1,pma->tilt_index_reduced,pma->has_tilt_reduced)){
        if(matter_is_index(index_tilt2,pma->tilt_index_normal,pma->has_tilt_normal)){
          current_tilt_offset = 2.0;
        }
        else if(matter_is_index(index_tilt2,pma->tilt_index_reduced,pma->has_tilt_reduced)){
          current_tilt_offset = 4.0;
        }
        else{
          class_stop(pma->error_message,"Tilt index %i not recognized",index_tilt2);
        }
      }
      else{
        class_stop(pma->error_message,"Tilt index %i not recognized",index_tilt1);
      }
      if(pma->matter_verbose > MATTER_VERBOSITY_RANGES){
        printf(" -> Found nu_real = %f (offset from bias : %f) \n",pma->bias-current_tilt_offset,current_tilt_offset);
      }
      pma->nu_real[index_tilt1_tilt2]=pma->bias-current_tilt_offset;
    }
    //End tilt2
  }
  //End tilt1
  /**
   * The imaginary part of the coefficients
   *   depends only on the coefficient index
   * */
  class_alloc(pma->nu_imag,
              pma->size_fft_result*sizeof(double),
              pma->error_message);
  for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
    /**
     * The factor of (N-1)/N might at first seem confusing, but it is necessary and mathematically correct:
     *
     * For any FFT, we want factors of exp(2*pi*i*m*n)
     *  In our case, the FFT goes over log(k),
     *  which was sampled as k_m = k_0 * exp(m/(N-1)*dkap)
     * (where dkap = log(k_max)-log(k_min)
     *  Notice the N-1. This is included to let k_(N-1) = k_max
     *
     * However, this (N-1) factor is not the one required by the FFT exponential
     *  The (N-1)/N is a sort of correction for this fact
     *
     * For this, let us calculate k_m*k^(nu_imag_n)
     *  k_m = k_0 * exp(m/(N-1)*dkap)
     *  k^(nu_imag_n) = exp(2*pi*n/dkap *(N-1)/N)
     *
     * =>
     *  k_m k^(nu_imag_n) = k_0*exp(2*pi*m*n/N)
     * Which is exactly of the form we want
     *
     * It was very important here, that the N-1 factor should cancel,
     *  which is only possible if we include this correction factor here
     * */
    pma->nu_imag[index_coeff]=_TWOPI_*(((double)(index_coeff))/(pma->deltalogk))*((double)pma->size_fft_input-1)/((double)pma->size_fft_input);
  }
  //End coeff
  return _SUCCESS_;
}

/**
 * Obtain sampling for the l values
 *
 * @param ppr        Input: pointer to precision structure
 * @param pth        Input: pointer to thermo structure
 * @param ppt        Input: pointer to perturbation structure
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_obtain_l_sampling(
                            struct matters * pma
                            ){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Obtaining l sampling\n");
  }
  int index_l;
  int current_l;
  int increment;
  int l_min = 2;
  int l_max = pma->l_lss_max;

  //The smallest stepsize is 1, so we can safely assume the maximum size being l_max
  class_alloc(pma->l_sampling,
              l_max*sizeof(double),
              pma->error_message);
  if(!pma->uses_all_l_sampling){
    /**
     * This is the normal logarithmic sampling that you will also
     *   see in other parts of class, like e.g. the spectra module
     *
     *
     * We start from l = 2 and increase it with a logarithmic step
     * */

    index_l = 0;
    current_l = l_min;
    increment = MAX((int)(current_l * (pow(pma->l_logstep,pma->angular_rescaling)-1.)),1);
    pma->l_sampling[index_l]=current_l;

    while (((current_l+increment) < l_max) &&
           (increment < pma->l_linstep*pma->angular_rescaling)) {

      index_l ++;
      current_l += increment;
      pma->l_sampling[index_l]=current_l;

      increment = MAX((int)(current_l * (pow(pma->l_logstep,pma->angular_rescaling)-1.)),1);

    }

    /**
     * When the logarithmic step becomes larger than some linear step,
     * stick to this linear step until we reach l_max
     * */

    increment = MAX((int)(pma->l_linstep*pma->angular_rescaling+0.5),1);

    while ((current_l+increment) <= l_max) {

      index_l ++;
      current_l += increment;
      pma->l_sampling[index_l]=current_l;

    }

    /**
     * The last value has to be set to exactly l_max
     *  (Otherwise there would be out-of-bounds problems with splining)
     * Of course we only need to add the additonal
     *  value of l_max, if we don't already hit it
     *  by accident
     * */

    if (current_l != l_max) {

      index_l ++;
      current_l = l_max;
      pma->l_sampling[index_l]=current_l;

    }

    pma->l_size = index_l+1;
    class_realloc(pma->l_sampling,
                  pma->l_sampling,
                  (index_l+1)*sizeof(double),
                  pma->error_message);
  }
  else{
    /**
     * The l_size is l_max-1,
     *  thus index_l is smaller or equal to l_max-2
     *  thus index_l+2 is smaller or equal to l_max,
     * as desired
     * */
    pma->l_size = l_max-1;
    for(index_l=0;index_l<pma->l_size;++index_l){
      pma->l_sampling[index_l]=index_l+2;
    }
    class_realloc(pma->l_sampling,
                  pma->l_sampling,
                  pma->l_size*sizeof(double),
                  pma->error_message);
  }
  pma->l_size_recursion = (int)(pma->uses_bessel_storeall?(pma->l_sampling[pma->l_size-1]+1):pma->l_size);
  return _SUCCESS_;
}


/**
 * Obtain sampling for the t values (the values of relative distance along line of sight)
 *
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_obtain_t_sampling(struct matters* pma){
/**
   * We want to do a normal integration,
   *  if we do not use limber approximation
   * Otherwise we only have the value t=1
   * */
  if(!pma->uses_limber_approximation){
    class_alloc(pma->t_sampling,
                pma->t_size*sizeof(double),
                pma->error_message);
    class_alloc(pma->t_weights,
                pma->t_size*sizeof(double),
                pma->error_message);
    class_call(array_weights_gauss_limits(
                                    pma->t_sampling,
                                    pma->t_weights,
                                    0.0,//1e-8 for trapz integration
                                    1.0,
                                    pma->t_size,
                                    gauss_type_legendre,//gauss_type_legendre_half,//gauss_type_trapezoid,//gauss_type_legendre,
                                    pma->error_message),
              pma->error_message,
              pma->error_message);
  }
  else{
    class_alloc(pma->t_sampling,
                sizeof(double),
                pma->error_message);
    class_alloc(pma->t_weights,
                sizeof(double),
                pma->error_message);
    pma->t_size=1;
    pma->t_sampling[0]=1.0;
    pma->t_weights[0]=1.0;
  }
  /**
   * If we want to obtain f_n^{ij}(t) for a subset of
   *  t values and spline it for all others,
   *  we can create a seperate sampling (t_spline_sampling)
   * Since we never integrate over that sampling,
   * we ingore the weights returned from this method
   *
   * Otherwise t spline and t are the same
   * */
  if(pma->uses_intxi_interpolation){
    double* ignore;
    class_alloc(ignore,
                pma->t_spline_size*sizeof(double),
                pma->error_message);
    class_alloc(pma->t_spline_sampling,
                pma->t_spline_size*sizeof(double),
                pma->error_message);
    class_call(array_weights_gauss_limits(
                                    pma->t_spline_sampling,
                                    ignore,
                                    0.0,//+1e-8 ? for trapz integration
                                    1.0,
                                    pma->t_spline_size,
                                    gauss_type_trapezoid,//gauss_type_trapezoid,//gauss_type_legendre,
                                    pma->error_message),
              pma->error_message,
              pma->error_message);
    double a = 0.75;//0.75;lens//1.0;//0.5;
    int index_t;
    for(index_t=0;index_t<pma->t_spline_size;++index_t){
      double t = pma->t_spline_sampling[index_t];
      pma->t_spline_sampling[index_t]=(1.-a)*t+a*(2.*t-t*t);
    }
    free(ignore);
  }
  else{
    pma->t_spline_size = pma->t_size;
    pma->t_spline_sampling = pma->t_sampling;
  }
  //Ifend xi interpolation
  return _SUCCESS_;
}

/**
 * Interpolate a prepared window
 *
 * @param pma        Input: pointer to matter structure
 * @param tau        Input: value of conformal time
 * @param index_ic   Input: index of initial condition
 * @param index_radtp Input: index of radial type
 * @param index_wd   Input: index of window type
 * @param last       Input/Output: last succesful interpolation
 * @param derivative_type Input: type of bessel derivative
 * @param win_val    Output: value of window function
 * @return the error status
 */
int matter_get_prepared_window_at(
                          struct matters* pma,
                          double tau,
                          int index_ic,
                          int index_radtp,
                          int index_wd,
                          int* last,
                          int derivative_type,
                          double* win_val
                          ){
  double a,b,h;
  index_ic = 0;
  if(derivative_type>=0){
    if(tau<pma->ptw_sampling[index_wd*pma->ptw_size] || tau>pma->ptw_sampling[(index_wd+1)*pma->ptw_size-1]){
      *win_val = 0.0;
      return _SUCCESS_;
    }

    class_call(matter_spline_hunt(pma->ptw_sampling+index_wd*pma->ptw_size,
                                  pma->ptw_size,
                                  tau,
                                  last,
                                  &h,
                                  &a,
                                  &b,
                                  pma->error_message),
                 pma->error_message,
                 pma->error_message);
    int last_index = *last;

    if(derivative_type==0){
      *win_val = a*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                +b*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1];
    }
    if(derivative_type==1){
      *win_val = a*pma->ptw_dwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                +b*pma->ptw_dwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1];
    }
    if(derivative_type==2){
      *win_val = a*pma->ptw_ddwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                +b*pma->ptw_ddwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1];
    }
    if(derivative_type==3){
      class_stop(pma->error_message,"Currently this is not implemented ...");
      /* *win_val = a*(
                    -pma->ptw_ddwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                    -2.0/(pma->tau0-tau)*pma->ptw_dwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                    +(l*(l+1.0)-2.0)*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]/(pma->tau0-tau)/(pma->tau0-tau)
                    )
                +b*(
                    -pma->ptw_ddwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1]
                    -2.0/(pma->tau0-tau)*pma->ptw_dwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1]
                    +(l*(l+1.0)-2.0)*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1]/(pma->tau0-tau)/(pma->tau0-tau)
                  );*/
    }
    if(derivative_type==4){
      *win_val = a*(
                    -pma->ptw_ddwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                    -2.0/(pma->tau0-tau)*pma->ptw_dwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]
                    -2.0*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index]/(pma->tau0-tau)/(pma->tau0-tau)
                    )
                +b*(
                    -pma->ptw_ddwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1]
                    -2.0/(pma->tau0-tau)*pma->ptw_dwindow[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1]
                    -2.0*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_size+last_index+1]/(pma->tau0-tau)/(pma->tau0-tau)
                    );
    }
  }
  else if(derivative_type==-1){
     if(tau<pma->ptw_integrated_sampling[index_wd*pma->ptw_integrated_size] ||
        tau>pma->ptw_integrated_sampling[index_wd*pma->ptw_integrated_size+pma->ptw_integrated_size-1]){
      *win_val = 0.0;
      return _SUCCESS_;
    }

    class_call(matter_spline_hunt(pma->ptw_integrated_sampling+index_wd*pma->ptw_integrated_size,
                                  pma->ptw_integrated_size,
                                  tau,
                                  last,
                                  &h,
                                  &a,
                                  &b,
                                  pma->error_message),
                 pma->error_message,
                 pma->error_message);
    int last_index = *last;

    *win_val = a*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_integrated_size+last_index]
              +b*pma->ptw_window[index_ic*pma->radtp_size_total+index_radtp][index_wd*pma->ptw_integrated_size+last_index+1];
  }
  return _SUCCESS_;
}


/**
 * Obtain the indices
 *
 * @param ppm        Input: pointer to primordial structure
 * @param ppt        Input: pointer to perturbs structure
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_obtain_indices(
                          struct matters* pma
                          ){
  if(pma->matter_verbose> MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Obtaining indices (source types and perturbation types) \n");
  }

  /**
   * Setting flags of which types are included
   * */
  pma->has_cltp_nc = _TRUE_;//pma->has_cl_number_count;
  pma->has_stp_delta_m = _TRUE_;//pma->has_nc_density;
  pma->has_redshift_space_distortion = _FALSE_;//pma->has_nc_rsd;
  pma->has_lensing_terms = _FALSE_;//pma->has_nc_lens;
  pma->has_gravitational_terms = _FALSE_;//pma->has_nc_gr;
  pma->has_doppler_terms = _FALSE_;//pma->has_redshift_space_distortion;

  pma->has_cltp_sh = _TRUE_;//pma->has_cl_lensing_potential;
  pma->has_cl_shear = _TRUE_;//pma->has_cltp_sh;


  pma->has_integrated_windows = (pma->has_cl_shear || pma->has_lensing_terms || pma->has_gravitational_terms);
  pma->has_unintegrated_windows = (pma->has_cltp_nc && (pma->has_stp_delta_m || pma->has_redshift_space_distortion || pma->has_gravitational_terms || pma->has_doppler_terms));
  pma->has_stp_phi_plus_psi = (pma->has_lensing_terms || pma->has_gravitational_terms || pma->has_cl_shear);

  pma->has_window_differentiation = (pma->has_redshift_space_distortion || (pma->uses_density_splitting && pma->has_stp_delta_m));

  /**
   * Defining indices of source types
   *
   * Here we explicitly give the correspondence between the types
   *  used in the perturbations module and the matter module
   * */
  int radtp_size_counter = 0;

  /**
   * Defining indices of radial types or radtps
   *
   * These correspond to a combination of
   *   1) the source type
   *   2) the bessel function type
   *
   * When different relations are used,
   *  it is possible to combine some of these
   *  or instead to split them up
   * */
  if(pma->uses_density_splitting){
    class_define_index(pma->radtp_dens1,          pma->has_stp_delta_m,radtp_size_counter,                                 1);
    class_define_index(pma->radtp_dens2,          pma->has_stp_delta_m,radtp_size_counter,                                 1);
  }else{
    class_define_index(pma->radtp_dens,           pma->has_stp_delta_m,radtp_size_counter,                                 1);
  }
  if(pma->uses_rsd_combination){
    class_define_index(pma->radtp_rsd_combined,   pma->has_redshift_space_distortion,radtp_size_counter,                   1);
  }
  else{
    class_define_index(pma->radtp_rsd,            pma->has_redshift_space_distortion,radtp_size_counter,                   1);
    class_define_index(pma->radtp_dop1,           pma->has_redshift_space_distortion,radtp_size_counter,                   1);
    class_define_index(pma->radtp_dop2,           pma->has_redshift_space_distortion,radtp_size_counter,                   1);
  }
  class_define_index(pma->radtp_g1,               pma->has_gravitational_terms,radtp_size_counter,                         1);
  class_define_index(pma->radtp_g2,               pma->has_gravitational_terms,radtp_size_counter,                         1);
  class_define_index(pma->radtp_g3,               pma->has_gravitational_terms,radtp_size_counter,                         1);
  class_define_index(pma->radtp_nclens,           pma->has_lensing_terms,radtp_size_counter,                               1);
  class_define_index(pma->radtp_shlens,           pma->has_cl_shear,radtp_size_counter,                                    1);
  class_define_index(pma->radtp_g4,               pma->has_gravitational_terms,radtp_size_counter,                         1);
  class_define_index(pma->radtp_g5,               pma->has_gravitational_terms,radtp_size_counter,                         1);

  pma->radtp_size_total = radtp_size_counter;
  pma->radtp_grid_size = pma->radtp_size_total*pma->radtp_size_total;

  /**
   * Finally we want to count the total number of
   * Cl - types like number-count Cl's (nCl/dCl) or shear Cl's (sCl)
   * Currently only nCl and sCl are supported
   * */
  int cltp_size_counter = 0;
  class_define_index(pma->cltp_index_nc,pma->has_cltp_nc,cltp_size_counter,                 1);
  class_define_index(pma->cltp_index_sh,pma->has_cltp_sh,cltp_size_counter,                 1);
  pma->cltp_size = cltp_size_counter;
  pma->cltp_grid_size = (pma->cltp_size*(pma->cltp_size+1))/2;

  if(pma->matter_verbose>MATTER_VERBOSITY_INDICES){
    printf(" -> Found requested Cl's :\n");
    if(pma->has_cltp_nc){
      printf("    -> Number Count Cl's \n");
    }
    if(pma->has_cltp_sh){
      printf("    -> Shear Cl's \n");
    }
  }
  return _SUCCESS_;
}


/**
 * Obtain the indices for the bessel integrals
 *
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_obtain_bi_indices(
                              struct matters* pma
                              ){
  if(pma->matter_verbose >MATTER_VERBOSITY_FUNCTIONS ){
    printf("Method :: Obtaining bessel integral and tilt indices \n");
  }
  /**
   * Whereas before we found how the sources relate to each other
   *  (Cosmological part)
   * now we find how the bessel integrals and tilts are related
   *  (Geometrical part)
   * Of course sometimes we don't need to calculate all possible
   *  tilts of the bessel integrals, and this depends on not only
   *  our options, but also if the corresponding radial types are
   *  defined or not
   *
   * */
  int bi_size_counter   = 0;
  int tilt_size_counter = 0;
  int bi_normal_size_counter =0;
  int bi_reduced_size_counter=0;
  int bi_lfactor_size_nc_counter=0;
  int bi_lfactor_size_sh_counter=0;

  /**
   * Of course checking how many functions and which are really required
   *  does get a bit tedious at times
   * */
  pma->has_bitp_normal = _FALSE_;
  pma->has_bitp_nu_reduced = _FALSE_;
  pma->has_bitp_lfactor = _FALSE_;
  if(pma->uses_relative_factors){
    if(pma->has_redshift_space_distortion || pma->has_gravitational_terms || pma->has_lensing_terms){
      bi_reduced_size_counter=1;
    }
  }
  {
    if(pma->uses_density_splitting){
      pma->has_bitp_nu_reduced = _TRUE_;
      pma->has_bitp_lfactor = _TRUE_;
      if(!pma->uses_relative_factors){
        bi_reduced_size_counter++;
      }
      bi_lfactor_size_nc_counter++;
    }
    else{
      pma->has_bitp_normal = _TRUE_;
      bi_normal_size_counter++;
    }
  }
  if(pma->has_redshift_space_distortion){
    if(!pma->uses_relative_factors){
      if(pma->uses_rsd_combination){
        bi_reduced_size_counter++;
      }
      else{
        bi_reduced_size_counter+=3;
      }
    }
    pma->has_bitp_nu_reduced = _TRUE_;
  }
  if(pma->has_lensing_terms){
    bi_lfactor_size_nc_counter++;
    pma->has_bitp_lfactor = _TRUE_;
  }
  if(pma->has_cl_shear){
    bi_lfactor_size_sh_counter++;
    pma->has_bitp_lfactor = _TRUE_;
  }
  if(pma->has_gravitational_terms){
    if(!pma->uses_relative_factors){
      bi_reduced_size_counter+=5;
    }
    else{
      bi_reduced_size_counter+=2;
    }
    pma->has_bitp_nu_reduced= _TRUE_;
  }
  pma->has_tilt_normal = pma->has_bitp_normal;
  pma->has_tilt_reduced = (pma->has_bitp_lfactor || pma->has_bitp_nu_reduced);

  /**
   * Once we have find which tilts should exist
   *  and which bessel integral types should exist,
   *  we now define corresponding indices
   * */
  // Normal bessel functions
  class_define_index(pma->bitp_index_normal, pma->has_bitp_normal,bi_size_counter,           1);
  class_define_index(pma->tilt_index_normal, pma->has_tilt_normal,tilt_size_counter,         1);
  // Nu-2 and Nu-4 bessel functions
  class_define_index(pma->bitp_index_nu_reduced, pma->has_bitp_nu_reduced ,bi_size_counter,  1);
  class_define_index(pma->tilt_index_reduced, pma->has_tilt_reduced,tilt_size_counter,    1);
  // l(l+1) prefactor of bessel functions
  // (does not introduce new tilt)
  class_define_index(pma->bitp_index_lfactor, pma->has_bitp_lfactor,bi_size_counter,         1);

  pma->bitp_size = bi_size_counter;
  pma->tilt_size = tilt_size_counter;
  pma->tilt_grid_size = (pma->tilt_size*(pma->tilt_size+1))/2;


  /**
   * Now we once again build an analogy table,
   *  saying for each bessel integral types
   *  which radial types can be found.
   *
   * We also define a small macro that takes care of
   *  checking whether all desired indices have been correctly assigned
   * This is mostly a macro checking whether or not the function is
   *  written correctly is changed
   * */
  if(pma->matter_verbose>MATTER_VERBOSITY_INDICES){
    printf(" -> Analysis of radial and bessel integral type structure : \n");
    printf(" -> Found number of tilts %i (symmetric grid %i) \n", pma->tilt_size,pma->tilt_grid_size);
    printf(" -> Found number of bessel integral types %i \n",pma->bitp_size);
    printf("    -> Bitp normal is %sfound (%4d indices)\n",(pma->has_bitp_normal?"":"not "),bi_normal_size_counter);
    printf("    -> Bitp reduced is %sfound (%4d indices)\n",(pma->has_bitp_nu_reduced?"":"not "),bi_reduced_size_counter);
    printf("    -> Bitp lfactor is %sfound (%4d indices(nc only), %4d indices(sh only))\n",(pma->has_bitp_lfactor?"":"not "),bi_lfactor_size_nc_counter,bi_lfactor_size_sh_counter);
  }
  class_alloc(pma->radtps_of_bitp,
              pma->bitp_size*pma->cltp_size*sizeof(double*),
              pma->error_message);
  class_alloc(pma->radtp_of_bitp_size,
              pma->bitp_size*pma->cltp_size*sizeof(double),
              pma->error_message);

  #define matter_index_correspondence(store_array,condition,index_in_array_counter,original_index) \
  if((condition)){                                                                                 \
    (store_array)[--(index_in_array_counter)] = (original_index);                                  \
  }
  if(pma->has_bitp_normal){
    if(pma->has_cltp_nc){
      class_alloc(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_normal],
                  bi_normal_size_counter*sizeof(double),
                  pma->error_message);
      pma->radtp_of_bitp_size[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_normal] = bi_normal_size_counter;
      matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_normal],(pma->has_stp_delta_m  && (!pma->uses_density_splitting)),bi_normal_size_counter,pma->radtp_dens)
      class_test(bi_normal_size_counter!=0,
                 pma->error_message,
                 "Number of radial types for bessel integral type 'normal' do not match up.");
    }
    if(pma->has_cltp_sh){
      pma->radtp_of_bitp_size[pma->cltp_index_sh*pma->bitp_size+pma->bitp_index_normal] = 0;
    }
  }
  if(pma->has_bitp_nu_reduced){
    if(pma->has_cltp_nc){
      class_alloc(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],
                  bi_reduced_size_counter*sizeof(double),
                  pma->error_message);
      pma->radtp_of_bitp_size[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced] = bi_reduced_size_counter;
      if(pma->uses_relative_factors){
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->uses_relative_factors,bi_reduced_size_counter,pma->radtp_combined)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g4)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g5)
      }
      else{
        if(pma->uses_rsd_combination){
          matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_redshift_space_distortion,bi_reduced_size_counter,pma->radtp_rsd_combined)
        }
        else{
          matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_redshift_space_distortion,bi_reduced_size_counter,pma->radtp_rsd)
          matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_redshift_space_distortion,bi_reduced_size_counter,pma->radtp_dop1)
          matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_redshift_space_distortion,bi_reduced_size_counter,pma->radtp_dop2)
        }
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],(pma->has_stp_delta_m && pma->uses_density_splitting),bi_reduced_size_counter,pma->radtp_dens1)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g1)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g2)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g3)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g4)
        matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_nu_reduced],pma->has_gravitational_terms,bi_reduced_size_counter,pma->radtp_g5)
      }
      class_test(bi_reduced_size_counter!=0,
                 pma->error_message,
                 "Number of radial types for bessel integral type 'reduced' do not match up.");
    }
    if(pma->has_cltp_sh){
      pma->radtp_of_bitp_size[pma->cltp_index_sh*pma->bitp_size+pma->bitp_index_nu_reduced] = 0;
    }
  }
  if(pma->has_bitp_lfactor){
    if(pma->has_cltp_nc){
      class_alloc(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_lfactor],
                  bi_lfactor_size_nc_counter*sizeof(double),
                  pma->error_message);
      pma->radtp_of_bitp_size[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_lfactor] = bi_lfactor_size_nc_counter;
      matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_lfactor],pma->has_lensing_terms,bi_lfactor_size_nc_counter,pma->radtp_nclens)
      matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_nc*pma->bitp_size+pma->bitp_index_lfactor],(pma->has_stp_delta_m && pma->uses_density_splitting),bi_lfactor_size_nc_counter,pma->radtp_dens2)
      class_test(bi_lfactor_size_nc_counter!=0,
                 pma->error_message,
                 "Number of radial types for bessel integral type 'lfactor' do not match up.");
    }
    if(pma->has_cltp_sh){
      class_alloc(pma->radtps_of_bitp[pma->cltp_index_sh*pma->bitp_size+pma->bitp_index_lfactor],
                  bi_lfactor_size_sh_counter*sizeof(double),
                  pma->error_message);
      pma->radtp_of_bitp_size[pma->cltp_index_sh*pma->bitp_size+pma->bitp_index_lfactor] = bi_lfactor_size_sh_counter;
      matter_index_correspondence(pma->radtps_of_bitp[pma->cltp_index_sh*pma->bitp_size+pma->bitp_index_lfactor],pma->has_cl_shear,bi_lfactor_size_sh_counter,pma->radtp_shlens)
      class_test(bi_lfactor_size_sh_counter!=0,
                 pma->error_message,
                 "Number of radial types for bessel integral type 'lfactor' do not match up.");
    }
  }
  /**
   * Finally we can print what types were found
   *   and what bessel integral types they correspond to
   * */
  if(pma->matter_verbose > MATTER_VERBOSITY_INDICES){
    int i,j,k;
    for(k=0;k<pma->cltp_size;++k){
      if( (!matter_is_index(k,pma->cltp_index_nc,pma->has_cltp_nc))
         &&(!matter_is_index(k,pma->cltp_index_sh,pma->has_cltp_sh))
        ){continue;}
      printf(" -> Searching for correspondences at cltp %i \n",k);
      for(i=0;i<pma->bitp_size;++i){
        for(j=0;j<pma->radtp_of_bitp_size[k*pma->bitp_size+i];++j){
          if(pma->uses_relative_factors){
            if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_combined,pma->uses_relative_factors)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'combined' \n");
              }
            }
          }
          else{
            if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_dens,pma->has_stp_delta_m && (!pma->uses_density_splitting))){
              if(matter_is_index(i,pma->bitp_index_normal,pma->has_bitp_normal)){
                printf("    -> Found in bitp 'normal' index 'density' \n");
              }
              else if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'density' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_dens1,pma->has_stp_delta_m && pma->uses_density_splitting)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'density (part 1)' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_dens2,pma->has_stp_delta_m && pma->uses_density_splitting)){
              if(matter_is_index(i,pma->bitp_index_lfactor,pma->has_bitp_lfactor)){
                printf("    -> Found in bitp 'l(l+1) factor' index 'density (part 2)' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_dop1,pma->has_redshift_space_distortion && (!pma->uses_rsd_combination))){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'doppler 1' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_dop2,pma->has_redshift_space_distortion && (!pma->uses_rsd_combination))){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'doppler 2' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_rsd,pma->has_redshift_space_distortion && (!pma->uses_rsd_combination))){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'RSD (dominant term)' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_rsd_combined,pma->has_redshift_space_distortion && (pma->uses_rsd_combination))){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'RSD (all combined)' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_g1,pma->has_gravitational_terms)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'gr 1' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_g2,pma->has_gravitational_terms)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'gr 2' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_g3,pma->has_gravitational_terms)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'gr 3' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_g4,pma->has_gravitational_terms)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'gr 4' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_g5,pma->has_gravitational_terms)){
              if(matter_is_index(i,pma->bitp_index_nu_reduced,pma->has_bitp_nu_reduced)){
                printf("    -> Found in bitp 'reduced' index 'gr 5' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_nclens,pma->has_cltp_nc && pma->has_lensing_terms)){
              if(matter_is_index(i,pma->bitp_index_lfactor,pma->has_bitp_lfactor)){
                printf("    -> Found in bitp 'l(l+1) factor' index 'number count lensing' \n");
              }
            }
            else if(matter_is_index(pma->radtps_of_bitp[k*pma->bitp_size+i][j],pma->radtp_shlens,pma->has_cl_shear && pma->has_cltp_sh)){
              if(matter_is_index(i,pma->bitp_index_lfactor,pma->has_bitp_lfactor)){
                printf("    -> Found in bitp 'l(l+1) factor' index 'shear lensing' \n");
              }
            }
            //Ifend select radtp
          }
          //Ifend uses relative factors
        }
        //End radtps
      }
      //End bitp
    }
    //End cltp
  }
  return _SUCCESS_;
}


/**
 * Obtain the bessel functions using recursion
 *
 * @param pma        Input: pointer to matter structure
 * @return the error status
 */
int matter_obtain_bessel_recursion_parallel(struct matters* pma){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS ){
    printf("Method :: Obtain bessel integrals from recursion \n");
  }

  //long long TOTAL_ALLOC;
#ifdef _OPENMP
  double start_bessel_omp = omp_get_wtime();
#else
  double start_bessel_omp = 0.0;
#endif
  int index_tilt1,index_tilt2,index_tilt1_tilt2;
  int index_coeff;
  int index_l,index_l_eval;
  int index_t;
  double y_max;
  double y_min;
  int bessel_recursion_l_size;

  /**
   * The maximum l of recursion that we need to reach
   *  is simply given by the last l value in l sampling
   * */
  double bessel_recursion_l_max = pma->l_sampling[pma->l_size-1];
  /**
   * There is a semi-analytical formula for calculating how
   *  many l values are required to reach a given accuracy
   *  of the bessel integrals.
   * This analytic formula simply explodes for t->1,
   *  which we 'catch' by putting an arbitrary, but very very large
   *  number here
   * It can happen that actually more than this number of l
   *  would be required, giving us higher errors
   * We assume however, that this case is VERY rare
   * */
  int back_complicated_max_size = 25000;
  /**
   * One of the more complicated methods can actually
   * recognize its own failure
   *  (It starts a backward recursion, that can be connected
   *   to the analytic limit for l=0)
   * This is the allowed deviation from the analytic limit
   * */
  double BI_ALLOWED_ERROR = 1e-6;
  /**
   * Some of the simpler recursion techniques become
   *  unsafe for small imaginary parts
   * Here we explicitly switch those off
   * */
  double NU_IMAG_BI_RECURSION_SWITCH = -1.0;//20.0;//20.0;
  /**
   * We want as many samples as the user requested, plus two additional ones:
   *  t = 1.0 and t = 0.0
   * */
  int bi_recursion_t_size = pma->bessel_recursion_t_size+2;
  /**
   * We are going to use a formula of the type
   *  1/(1-t), and rounding that to an integer
   * Of course not all doubles are representable as an integer,
   *  so we are going to do a simple cutoff procedure:
   * We are going to use
   *  1/(1-t+inv_maximum_representable_integer)
   * which can be at most
   *  1/inv_maximum_representable_integer
   *
   * Thus, if our maximum representable integer is
   *  2*10^9, we would set the flag to 0.5e-9
   *
   * However, just to be a bit more careful, we
   *  choose 1e-5
   * */
  double inv_maximum_representable_integer = 1e-5;
  double nu_min_forward_real = 1.5;
  double l_max_forward_real = 1200;
  double l_max_backward_real = 5000;
  /**
   * Now we can set the y_max and y_min variables
   *
   * The cases t=1.0 and t=0.0 are handled completely seperately anyway,
   * so we just need reasonable limits that are not 'too' far off,
   *
   * Here, we choose the immediate values from
   * (BI_MIN_T up to 1.0-BI_SAMPLING_EPSILON)
   * which is inside the interval
   * (0.0,1.0)
   * */
  double BI_SAMPLING_EPSILON = 1e-8;//1e-8
  double BI_MIN_T = 1e-6;//1e-6
  y_max = -log(BI_SAMPLING_EPSILON);
  y_min = -log(1.0-BI_MIN_T);

  /**
   * If our l is sampled logarithmically,
   * the same can not be done for the recursion relations
   * (which require every single l)
   *
   * As such, we can have a different size here
   * (bessel_recursion_l_size instead of pma->l_size)
   * */
  bessel_recursion_l_size = bessel_recursion_l_max+1;
  pma->l_size_recursion = (pma->uses_bessel_storeall?bessel_recursion_l_size:pma->l_size);

  /**
   * Allocate the arrays in which we want to store the final bessel integrals
   *  (Bessel Integrals are shortened to BI)
   * */
  class_alloc(pma->bi_real,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  class_alloc(pma->bi_imag,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  class_alloc(pma->bi_size,
              pma->tilt_grid_size*sizeof(int*),
              pma->error_message);
  class_alloc(pma->bi_max,
              pma->tilt_grid_size*sizeof(double*),
              pma->error_message);
  class_alloc(pma->bi_sampling,
              bi_recursion_t_size*sizeof(double),
              pma->error_message);
  /**
   * Define and allocate temporary arrays,
   *  which are required for the evaluations
   * */
  for(index_tilt1=0;index_tilt1<pma->tilt_size;++index_tilt1){
    for(index_tilt2=index_tilt1;index_tilt2<pma->tilt_size;++index_tilt2){
      index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);
      class_alloc(pma->bi_real[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      class_alloc(pma->bi_imag[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      class_alloc(pma->bi_size[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(int),
                  pma->error_message);
      class_alloc(pma->bi_max[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double),
                  pma->error_message);
      /**
       * Allocate the real and imaginary arrays
       * Also allocate the sampling array
       * */
      for(index_l=0;index_l<pma->l_size_recursion;++index_l){
        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
          class_alloc(pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                      bi_recursion_t_size*sizeof(double),
                      pma->error_message);
          class_alloc(pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                      bi_recursion_t_size*sizeof(double),
                      pma->error_message);
        }
      }

      if(pma->matter_verbose > MATTER_VERBOSITY_BESSEL){
        printf(" -> Obtaining recursion starting bessel integrals for tilt %f \n",pma->bias-pma->nu_real[index_tilt1_tilt2]);
      }
      int abort = _FALSE_;
      #pragma omp parallel private(index_l,index_t,index_coeff) firstprivate(y_min,y_max,pma,bi_recursion_t_size,bessel_recursion_l_size,back_complicated_max_size,index_tilt1_tilt2)
      {
        double* max_t;
        double* abi_real;
        double* abi_imag;
        double* initial_abs;
        class_alloc_parallel(max_t,
                    bessel_recursion_l_size*sizeof(double),
                    pma->error_message);
        class_alloc_parallel(abi_real,
                    (bessel_recursion_l_size+back_complicated_max_size)*sizeof(double),
                    pma->error_message);
        class_alloc_parallel(abi_imag,
                    (bessel_recursion_l_size+back_complicated_max_size)*sizeof(double),
                    pma->error_message);
        class_alloc_parallel(initial_abs,
                    bessel_recursion_l_size*sizeof(double),
                    pma->error_message);

        double back_simple_time = 0;
        double for_simple_time = 0;
        double complex_time = 0;
        double inverse_time = 0;
        double taylor_time = 0;
        #pragma omp for schedule(dynamic,(pma->size_fft_result>=CHUNK_SIZE*omp_get_max_threads()?CHUNK_SIZE:1))
        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){

          if(pma->matter_verbose > MATTER_VERBOSITY_BESSEL){
            if(!MATTER_REWRITE_PRINTING){
              printf(" -> Obtaining bessel from nu[%3d/%3d] = %.10e+%.10ej \n",index_coeff,pma->size_fft_result-1,pma->nu_real[index_tilt1_tilt2],pma->nu_imag[index_coeff]);
            }
            else{
              printf("\r -> Obtaining bessel from nu[%3d/%3d] = %.10e+%.10ej ",index_coeff,pma->size_fft_result-1,pma->nu_real[index_tilt1_tilt2],pma->nu_imag[index_coeff]);
              fflush(stdout);
            }
          }

          /**
           * Set the nu_real and nu_imag parameters
           *  (as shorthands for quicker writing)
           * */
          double nu_real = pma->nu_real[index_tilt1_tilt2];
          double nu_imag = pma->nu_imag[index_coeff];

          /**
           * Obtain the initial bessel integrals and enter them already into the final array
           * */
          bessel_integral_recursion_initial_abs(bessel_recursion_l_max,nu_real,nu_imag,abi_real,abi_imag,initial_abs);

          for(index_l=0;index_l<pma->l_size_recursion;++index_l){
            index_l_eval = (pma->uses_bessel_storeall?index_l:(int)pma->l_sampling[index_l]);
            pma->bi_sampling[0]= 0.0;
            pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff][0] = abi_real[index_l_eval];
            pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff][0] = abi_imag[index_l_eval];
          }
          /**
           * Set the minimum t for which |I_l(nu,t)|<eps |I_l(nu,1)|
           *  conservatively to 0, and correct later
           * */

          memset(max_t,0,bessel_recursion_l_size*sizeof(double));
          /**
           * This flag is going to tell us which is the current
           * maximum l for any given t
           * that does not statisfy the exit condition:
           *  |I_l(nu,t)|<eps |I_l(nu,1)|
           *
           * The second term is given by initial_abs
           * */

          int l_max_cur = bessel_recursion_l_max;
          /**
           * This flag keeps track of overflows happening during the summation
           * of the hypergeometric functions. If no more overflows occur
           *  ( this flag being set to _FALSE_ )
           * then a simplified version of the summation can be used,
           * which does not check for further overflows.
           *
           * Always initialize as _TRUE_ !
           * */
          short overflow_flag = _TRUE_;
          for(index_t=1;index_t<bi_recursion_t_size;++index_t){
            /**
             * Obtain the t at which we want to sample
             * */
            double y = y_max-(y_max-y_min)*sqrt(((double)(index_t-1))/((double)(bi_recursion_t_size-2)));
            double t = 1.0-exp(-y);

            /**
             * Here are some semi-analytical determinations
             *  of the range to which each method extends
             *
             * Sadly, these only apply for a bias around ~1.9
             *  and for the maximum error of 1e-6
             *
             * Feel free to improve these formuli
             *
             * Forward simple:
             *  Limited to t close to 1,
             *  the deviation of which we fit,
             *  and capped off with a maximum value (that depends on nu_imag)
             *
             * Backward simple:
             *  Limited to t close to 1 (surprisingly),
             *  but with a much broader range
             * Update: Using the overflow-safe version, which requires the
             *  overflow_flag to keep track of overflows during calculation
             *
             * Self inverse taylor:
             *  Limited to very very close to 1,
             *  (otherwise too slow)
             *
             * Complex:
             *  Decides on forward or backward recursion
             *  using matrix invertibility criteria,
             *  but requires sufficient l to shave off initial errors in the
             *  starting conditions
             * */
            double forward_simple_factor_high_real = (nu_imag*nu_imag/10000.0+nu_imag/3.3+4.0);
            double forward_simple_factor_low_real = nu_imag/4.0*0.95;
            double forward_simple_factor = (forward_simple_factor_high_real-forward_simple_factor_low_real)*(nu_real+2.1)/4.0+forward_simple_factor_low_real;

            double forward_l_max_const = 3000.0+nu_imag*40.0;
            int l_max_forward_simple = (int)MIN((forward_simple_factor/(1-t+inv_maximum_representable_integer)),forward_l_max_const);

            double backward_simple_factor = 10.0+nu_imag/5.0;
            int l_max_backward_simple = (int)(backward_simple_factor/(1-t+inv_maximum_representable_integer));

            double self_inverse_taylor_factor = 15.0;
            int l_max_self_inverse_taylor = (int)(self_inverse_taylor_factor/(1-t+inv_maximum_representable_integer));

            int delta_l_required = (int)MIN((12./(1-t)),back_complicated_max_size);

            double backward_simple_lfactor = MAX(1.5-nu_imag/15,0.0);
            if(t<T_MIN_TAYLOR){
#ifdef _OPENMP
              double func_start_t = omp_get_wtime();
#else
              double func_start_t = 0.0;
#endif
              bessel_integral_recursion_taylor(l_max_cur,nu_real,nu_imag,t,max_t,initial_abs,abi_real,abi_imag);
#ifdef _OPENMP
              taylor_time += omp_get_wtime()-func_start_t;
#else
              taylor_time +=0.0;
#endif
            }
            /**
             * We want the hypergeometric series to have only very vew terms
             * The arguments are approximately l^2/4 * (1-z)^2/(1+z)^2 << 1
             *  (neglecting any nu dependence)
             * Using z = t^2 = (1-eps)^2 ~ 1-2eps
             *  we find (1-z)^2/(1+z)^2 ~ eps^2
             * Then we find (l*eps/2)^2 << 1
             *  As such we get eps << 2/l => 1-t = alpha*2/l with alpha<<1
             *  We find quick convergence for 1-t = 2*alpha/l < 2*T_MIN_INVERSE_TAYLOR/l_max
             * ( l < l_max , alpha = T_MIN_INVERSE_TAYLOR)
             * */
            else if(l_max_self_inverse_taylor >l_max_cur && t>1.-T_MIN_INVERSE_TAYLOR){

#ifdef _OPENMP
              double func_start_t = omp_get_wtime();
#else
              double func_start_t = 0.0;
#endif
              bessel_integral_recursion_inverse_self(l_max_cur,nu_real,nu_imag,t,abi_real,abi_imag,max_t,initial_abs,pma->error_message);
#ifdef _OPENMP
              inverse_time += omp_get_wtime()-func_start_t;
#else
              inverse_time += 0.0;
#endif
            }
            else if(
              ( nu_imag>NU_IMAG_BI_RECURSION_SWITCH &&
              l_max_forward_simple > l_max_cur )
              || (nu_real>nu_min_forward_real && l_max_cur<l_max_forward_real)
            ){
#ifdef _OPENMP
              double func_start_t = omp_get_wtime();
#else
              double func_start_t = 0.0;
#endif
              class_call_parallel(bessel_integral_recursion_forward_simple(
                                                       l_max_cur,
                                                       nu_real,
                                                       nu_imag,
                                                       t,
                                                       abi_real,
                                                       abi_imag,
                                                       max_t,
                                                       initial_abs,
                                                       pma->error_message),
                         pma->error_message,
                         pma->error_message);
#ifdef _OPENMP
              for_simple_time += omp_get_wtime()-func_start_t;
#else
              for_simple_time += 0.0;
#endif
            }
            else if(
              (nu_imag>NU_IMAG_BI_RECURSION_SWITCH &&
              l_max_backward_simple > l_max_cur) ||
              (l_max_cur<l_max_backward_real)
            ){
#ifdef _OPENMP
              double func_start_t = omp_get_wtime();
#else
              double func_start_t = 0.0;
#endif
              class_call_parallel(bessel_integral_recursion_backward_simple_safe(
                                                       l_max_cur,
                                                       (1.1+backward_simple_lfactor)*l_max_cur,
                                                       nu_real,
                                                       nu_imag,
                                                       t,
                                                       abi_real,
                                                       abi_imag,
                                                       max_t,
                                                       initial_abs,
                                                       &overflow_flag,
                                                       pma->error_message),
                         pma->error_message,
                         pma->error_message);
#ifdef _OPENMP
              back_simple_time += omp_get_wtime()-func_start_t;
#else
              back_simple_time += 0.0;
#endif
            }
            else{
#ifdef _OPENMP
                double func_start_t = omp_get_wtime();
#else
                double func_start_t = 0.0;
#endif
                class_call_parallel(bessel_integral_recursion_complicated(l_max_cur,
                                                                 l_max_cur+delta_l_required-1,
                                                                 nu_real,
                                                                 nu_imag,
                                                                 t,
                                                                 BI_ALLOWED_ERROR,
                                                                 abi_real,
                                                                 abi_imag,
                                                                 max_t,
                                                                 initial_abs,
                                                                 pma->error_message),
                           pma->error_message,
                           pma->error_message);
#ifdef _OPENMP
              complex_time += omp_get_wtime()-func_start_t;
#else
              complex_time += 0.0;
#endif
            }
            /**
             * After obtaining the corresponding bessel integrals through recursion
             *  for this particular value for t,
             *  we need to store them within the storage arrays
             * */
            for(index_l=0;index_l<pma->l_size_recursion;++index_l){
              index_l_eval = (pma->uses_bessel_storeall?index_l:(int)pma->l_sampling[index_l]);
              if(index_l_eval>l_max_cur){continue;}
              pma->bi_sampling[index_t]= 1.0-t;
              pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff][index_t] = abi_real[index_l_eval];
              pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff][index_t] = abi_imag[index_l_eval];
              if(max_t[index_l_eval]>=t){
                /**
                 * If the condition |I_l(nu,t)|<eps*|I_l(nu,1)|
                 * is fulfilled, we do not need to evaluate this mode
                 * for any smaller values of t
                 *
                 * Thus this mode is 'exiting' our evaluation range
                 *
                 * We can also then reallocate the arrays reaching to exactly this point
                 * And also set the size and maximum evaluable point before 0
                 * */
                pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff] = index_t+1;
                pma->bi_max[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff] = 1.0 - t;
                //TODO :: figure out why parallel reallocation leads to segmentation faults
                /*class_realloc_parallel(pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                              pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                              pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                              pma->error_message);
                class_realloc_parallel(pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                              pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                              pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                              pma->error_message);
                class_realloc_parallel(pma->bi_sampling[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                              pma->bi_sampling[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                              pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                              pma->error_message);*/
              }
              //Ifend exit
            }
            //End l
            /**
             * If a mode has 'exited',
             *  we need to check what the current
             *  maximum mode to still be evolved is
             * */
            index_l=l_max_cur;
            while(index_l>0 && max_t[index_l]>=t){
              index_l--;
              l_max_cur--;
            }
          }
          //End t

          /**
           * The analytic solutions at l=0 and l=1
           *  continue to t=1 without reaching their criterion
           * (The l=0 goes towards a constant, the l=1 decreases too slowly to exit in most cases)
           * */
          if(pma->uses_bessel_storeall){
            pma->bi_size[index_tilt1_tilt2][0*pma->size_fft_result+index_coeff] = bi_recursion_t_size;
            pma->bi_max[index_tilt1_tilt2][0*pma->size_fft_result+index_coeff] = 0.0;
            pma->bi_size[index_tilt1_tilt2][1*pma->size_fft_result+index_coeff] = bi_recursion_t_size;
            pma->bi_max[index_tilt1_tilt2][1*pma->size_fft_result+index_coeff] = 0.0;
          }

        }
        //End coeff
        if(MATTER_REWRITE_PRINTING){
          printf("\r                                                                             \n");
        }

        free(initial_abs);
        free(max_t);
        free(abi_real);
        free(abi_imag);
      }
      //End parallel
      if (abort == _TRUE_) return _FAILURE_;
    }
    //End tilt2
  }
  //End tilt1
  /**
   * Delete temporary arrays
   * */
#ifdef _OPENMP
  double end_bessel_omp = omp_get_wtime();
#else
  double end_bessel_omp = 0.0;
#endif
  if(pma->matter_verbose > MATTER_VERBOSITY_TIMING) {
    printf(" -> Obtaining (recursion) Bessel Integrals took %f REAL seconds \n",end_bessel_omp-start_bessel_omp);
  }
  return _SUCCESS_;
}

/**
 * Obtain the nonlinearity factor, replacing the values of
 * fft_coeff_real, and fft_coeff_imag
 *
 * @param pma                  Input: pointer to matter struct
 * @param fft_coeff_real       Input/Output: fft coefficients (real)
 * @param fft_coeff_imag       Input/Output: fft coefficients (imag)
 * @return the error status
 */
int matter_obtain_nonseparability(
                               struct matters* pma,
                               double ** fft_real,
                               double ** fft_imag
                              ){
  if(pma->matter_verbose>MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Obtain non-scale-invariance factor\n");
  }
  double* fft_coeff_factor_real;
  double* fft_coeff_factor_imag;
  double* fft_coeff_real = *fft_real;
  double* fft_coeff_imag = *fft_imag;
  int index_tau1,index_tau2,index_tau1_tau2,index_coeff;
  class_alloc(fft_coeff_factor_real,
        pma->size_fft_input*(pma->tau_grid_size+1)*sizeof(double),
        pma->error_message);
  class_alloc(fft_coeff_factor_imag,
        pma->size_fft_input*(pma->tau_grid_size+1)*sizeof(double),
        pma->error_message);
  for(index_tau1=0;index_tau1<pma->tau_size;++index_tau1){
    for(index_tau2=0;index_tau2<pma->tau_size;++index_tau2){
      index_tau1_tau2 = index_tau2*pma->tau_size+index_tau1;
      for(index_coeff = 0; index_coeff < pma->size_fft_result;++index_coeff){
        fft_coeff_factor_real[index_tau1_tau2*pma->size_fft_input+index_coeff] =
          fft_coeff_real[index_tau1_tau2*pma->size_fft_input+index_coeff]/
          (pma->growth_factor_tau[index_tau1]*pma->growth_factor_tau[index_tau2]);
        fft_coeff_factor_imag[index_tau1_tau2*pma->size_fft_input+index_coeff] =
          fft_coeff_imag[index_tau1_tau2*pma->size_fft_input+index_coeff]/
          (pma->growth_factor_tau[index_tau1]*pma->growth_factor_tau[index_tau2]);
      }
      //End index_coeff
    }
    //End tau 1
  }
  //End tau 2
  free(fft_coeff_real);
  free(fft_coeff_imag);
  *fft_real = fft_coeff_factor_real;
  *fft_imag = fft_coeff_factor_imag;
  return _SUCCESS_;
}

/**
 * Spline the bessel integrals after having recursively found them
 *
 * @param pma                  Input: pointer to matter struct
 * @return the error status
 */
int matter_spline_bessel_integrals_recursion(
                                  struct matters * pma
                                  ) {
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Spline (recursion) bessel integrals\n");
  }
#ifdef _OPENMP
  double spline_start_omp = omp_get_wtime();
#else
  double spline_start_omp = 0.0;
#endif
  /**
   * Define indices and allocate arrays
   * */
  int index_coeff;
  int index_l;
  int index_tilt1,index_tilt2,index_tilt1_tilt2;
  class_alloc(pma->ddbi_real,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  class_alloc(pma->ddbi_imag,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  for(index_tilt1=0;index_tilt1<pma->tilt_size;++index_tilt1){
    for(index_tilt2=index_tilt1;index_tilt2<pma->tilt_size;++index_tilt2){
      index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);
      class_alloc(pma->ddbi_real[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      class_alloc(pma->ddbi_imag[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      for(index_l=0;index_l<pma->l_size_recursion;++index_l){
        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
          class_alloc(pma->ddbi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                      pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                      pma->error_message);
          class_alloc(pma->ddbi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                      pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                      pma->error_message);
        }
      }
      int abort = _FALSE_;
      #pragma omp parallel private(index_l,index_coeff) firstprivate(pma,index_tilt1_tilt2)
      {
      #pragma omp for
      for(index_l=0;index_l<pma->l_size_recursion;++index_l){
        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
          class_call_parallel(array_spline_table_columns(pma->bi_sampling,
                                                pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                                                pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                                                1,
                                                pma->ddbi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                                                _SPLINE_EST_DERIV_,
                                                pma->error_message),
                    pma->error_message,
                    pma->error_message);
          class_call_parallel(array_spline_table_columns(pma->bi_sampling,
                                                pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                                                pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                                                1,
                                                pma->ddbi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                                                _SPLINE_EST_DERIV_,
                                                pma->error_message),
                    pma->error_message,
                    pma->error_message);
        }
        //End coeff
      }
      //End l
      }
      if(abort == _TRUE_) {return _FAILURE_;}
    }
    //End tilt2
  }
  //End tilt1
#ifdef _OPENMP
  double spline_end_omp = omp_get_wtime();
#else
  double spline_end_omp = 0.0;
#endif
  if(pma->matter_verbose > MATTER_VERBOSITY_TIMING ){
    printf(" -> Splining bessel integrals (recursion) took %f REAL seconds \n",spline_end_omp-spline_start_omp);
  }
  return _SUCCESS_;
}


/**
 * Cubic hermite spline interpolation
 *
 * @param x_array              Input: pointer x array
 * @param x_size               Input: size x array
 * @param array                Input: pointer y array
 * @param array_splined        Input: pointer ddy array
 * @param y_size               Input: size y array
 * @param x                    Input: x to interpolate at
 * @param last_index           Input/Output: last index at which x was found
 * @param result               Output: pointer to y(x) value
 * @param errmsg               Output: the error message
 * @return the error status
 */
int matter_interpolate_spline_growing_hunt(
                double * x_array,
                int x_size,
                double * array, //[index_y*x_size+index_x]
                double * array_splined,
                int y_size,
                double x,
                int * last_index,
                double * result,
                ErrorMsg errmsg
               ) {
  double h,a,b;
  int index_y;
  int inf,sup;
  matter_spline_hunt(x_array,x_size,x,last_index,&h,&a,&b,errmsg);
  inf = *last_index;
  sup = inf+1;
  for (index_y=0; index_y<y_size; index_y++){
    *(result+index_y) =
      a * *(array+inf+x_size*index_y) +
      b * *(array+sup+x_size*index_y) +
      ((a*a*a-a)* *(array_splined+inf+x_size*index_y) +
       (b*b*b-b)* *(array_splined+sup+x_size*index_y))*h*h/6.;
  }
  return _SUCCESS_;
}

/**
 * FFTlog the perturbation sources in a parallelized fashion
 *
 * @param pba                  Input: pointer to background struct
 * @param ppt                  Input: pointer to perturbation struct
 * @param ppm                  Input: pointer to primordial struct
 * @param pma                  Input: pointer to matter struct
 * @param sampled_sources      Input: array of sampled source
 * @param prim_spec            Input: array of sampled primordial spectrum
 * @param fft_coeff_real       Output: the fft coefficients (real)
 * @param fft_coeff_imag       Output: the fft coefficients (imaginary)
 * @return the error status
 */
int matter_FFTlog_perturbation_sources_parallel(
                struct matters * pma,
                double ** fft_coeff_real_ptr,
                double ** fft_coeff_imag_ptr
                ) {
  double* fft_coeff_real;
  double* fft_coeff_imag;
  int N_threads;
  int index_tau1,index_tau2,index_tau1_tau2;
#ifdef _OPENMP
  if(!pma->uses_separability){
    N_threads = omp_get_max_threads();
  }
  else{
    N_threads=1;
  }
#else
  N_threads = 1;
#endif
  int n_thread;


  double** integrand1;
  double** integrand2;
  class_alloc(integrand1,
              N_threads*sizeof(double*),
              pma->error_message);
  class_alloc(integrand2,
              N_threads*sizeof(double*),
              pma->error_message);
  class_alloc(pma->FFT_plan,
              N_threads*sizeof(struct FFT_plan*),
              pma->error_message);
  for(n_thread=0;n_thread<N_threads;++n_thread){
    FFT_planner_init(pma->size_fft_input,&(pma->FFT_plan[n_thread]));
    class_alloc(integrand1[n_thread],
                pma->size_fft_input*sizeof(double),
                pma->error_message);
    class_alloc(integrand2[n_thread],
                pma->size_fft_input*sizeof(double),
                pma->error_message);
  }
  if(pma->uses_separability){
    if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
      printf("Method :: FFT only today\n");
    }
    int index_stp1,index_stp2,index_stp1_stp2;
    int index_ic1,index_ic2,index_ic1_ic2;
    int index_coeff;
    /**
     * First we allocate the integrands and the fft coefficients
     * */
    class_alloc(fft_coeff_real,
          2*pma->size_fft_input*sizeof(double),
          pma->error_message);
    class_alloc(fft_coeff_imag,
          2*pma->size_fft_input*sizeof(double),
          pma->error_message);
    /**
     * Now we iterate over all combinations of sources and do the FFT transformation
     * */
    {
      {
        int abort = _FALSE_;
        {
          {
            int tid=0;
            index_tau1=pma->tau_size-1;
            index_tau2=pma->tau_size-1;
            /**
             * There is a neat trick with FFT transformations for real inputs,
             * we can do two transformations at once.
             *  In this case, we should simply ignore the second part,
             *  since there is only one tau value (tau0),
             * It was easier to simply set the second integrand to 0 than rewriting
             *  the entire FFT functionality
             * */
            for(index_coeff=0;index_coeff<pma->size_fft_input;++index_coeff){
              integrand1[tid][index_coeff] = pma->sampled_sources[index_coeff*pma->tau_size+index_tau1]
                                        *pma->sampled_sources[index_coeff*pma->tau_size+index_tau2]
                                        *pow(pma->k_sampling[index_coeff]/pma->k_sampling[0],-pma->bias);
            }
            for(index_coeff=0;index_coeff<pma->size_fft_input;++index_coeff){
              integrand2[tid][index_coeff] = 0.0;
            }
            FFT_real_short_planned(integrand1[tid],
                                   integrand2[tid],
                                   fft_coeff_real,
                                   fft_coeff_imag,
                                   fft_coeff_real+1*pma->size_fft_input,
                                   fft_coeff_imag+1*pma->size_fft_input,
                                   pma->FFT_plan[tid]);
            /**
             * The coefficients we have calculated are not yet the final coefficients
             *  For these, we have to multiply with k0^(nu_imag)
             * */
            for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
              double exp_factor = pow(pma->k_sampling[0],-pma->bias);
              double phase = -log(pma->k_sampling[0])*pma->nu_imag[index_coeff];
              double exp_real = exp_factor*cos(phase);
              double exp_imag = exp_factor*sin(phase);
              double coeff_real = fft_coeff_real[index_coeff];
              double coeff_imag = fft_coeff_imag[index_coeff];
              double newcoeff_real = coeff_real*exp_real-coeff_imag*exp_imag;
              double newcoeff_imag = coeff_real*exp_imag+coeff_imag*exp_real;
              fft_coeff_real[index_coeff] = newcoeff_real/pma->size_fft_input;
              fft_coeff_imag[index_coeff] = newcoeff_imag/pma->size_fft_input;
            }
            //end coeffs
          }
          //End stp2
        }
        //End stp1
        if(abort == _TRUE_){return _FAILURE_;}
      }
      //end ic2
    }
    //end ic1
    if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
      printf(" -> Returning from FFTlog... \n");
    }
  }else{
    if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
      printf("Method :: FFT for every tau combination\n");
    }
    int index_coeff;
    /**
     * Allocate the integrands and the coefficient arrays
     * */
    class_alloc(fft_coeff_real,
          pma->size_fft_input*(pma->tau_grid_size+1)*sizeof(double),
          pma->error_message);
    class_alloc(fft_coeff_imag,
          pma->size_fft_input*(pma->tau_grid_size+1)*sizeof(double),
          pma->error_message);
    /**
     * Now we iterate over all combinations of sources and do the FFT transformation
     * */
    {
      {
        {
          {
            int abort = _FALSE_;
            #pragma omp parallel for collapse(2) private(index_tau1,index_tau2,index_tau1_tau2,index_coeff) firstprivate(fft_coeff_real,fft_coeff_imag)
            for(index_tau1=0;index_tau1<pma->tau_size;++index_tau1){
              for(index_tau2=0;index_tau2<pma->tau_size;index_tau2+=2){
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                index_tau1_tau2 = index_tau1*pma->tau_size+index_tau2;
                /**
                 * There is a neat trick with FFT transformations for real inputs,
                 * we can do two transformations at once.
                 * */
                for(index_coeff=0;index_coeff<pma->size_fft_input;++index_coeff){
                  integrand1[tid][index_coeff] = pma->sampled_sources[index_coeff*pma->tau_size+index_tau1]
                                            *pma->sampled_sources[index_coeff*pma->tau_size+index_tau2]
                                            *pow(pma->k_sampling[index_coeff]/pma->k_sampling[0],-pma->bias);
                }
                for(index_coeff=0;index_coeff<pma->size_fft_input;++index_coeff){
                  integrand2[tid][index_coeff] = pma->sampled_sources[index_coeff*pma->tau_size+index_tau1]
                                            *pma->sampled_sources[index_coeff*pma->tau_size+index_tau2+1]
                                            *pow(pma->k_sampling[index_coeff]/pma->k_sampling[0],-pma->bias);
                }
                FFT_real_short_planned(integrand1[tid],
                                       integrand2[tid],
                                       fft_coeff_real+index_tau1_tau2*pma->size_fft_input,
                                       fft_coeff_imag+index_tau1_tau2*pma->size_fft_input,
                                       fft_coeff_real+(index_tau1_tau2+1)*pma->size_fft_input,
                                       fft_coeff_imag+(index_tau1_tau2+1)*pma->size_fft_input,
                                       pma->FFT_plan[tid]);
                /**
                 * The coefficients we have calculated are not yet the final coefficients
                 *  For these, we have to multiply with k0^(nu_imag)
                 * */
                for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
                  double exp_factor = pow(pma->k_sampling[0],-pma->bias);
                  double phase = -log(pma->k_sampling[0])*pma->nu_imag[index_coeff];
                  double exp_real = exp_factor*cos(phase);
                  double exp_imag = exp_factor*sin(phase);
                  double coeff_real = fft_coeff_real[index_tau1_tau2*pma->size_fft_input+index_coeff];
                  double coeff_imag = fft_coeff_imag[index_tau1_tau2*pma->size_fft_input+index_coeff];
                  double newcoeff_real = coeff_real*exp_real-coeff_imag*exp_imag;
                  double newcoeff_imag = coeff_real*exp_imag+coeff_imag*exp_real;
                  fft_coeff_real[index_tau1_tau2*pma->size_fft_input+index_coeff] = newcoeff_real/pma->size_fft_input;
                  fft_coeff_imag[index_tau1_tau2*pma->size_fft_input+index_coeff] = newcoeff_imag/pma->size_fft_input;
                  coeff_real = fft_coeff_real[(index_tau1_tau2+1)*pma->size_fft_input+index_coeff];
                  coeff_imag = fft_coeff_imag[(index_tau1_tau2+1)*pma->size_fft_input+index_coeff];
                  newcoeff_real = coeff_real*exp_real-coeff_imag*exp_imag;
                  newcoeff_imag = coeff_real*exp_imag+coeff_imag*exp_real;
                  fft_coeff_real[(index_tau1_tau2+1)*pma->size_fft_input+index_coeff] = newcoeff_real/pma->size_fft_input;
                  fft_coeff_imag[(index_tau1_tau2+1)*pma->size_fft_input+index_coeff] = newcoeff_imag/pma->size_fft_input;
                }
                //end coeffs
              }
              //End tau2
            }
            //End tau1
            if(abort==_TRUE_){return _FAILURE_;}
          }
          //End stp1
        }
        //End stp2
      }
      //End ic2
    }
    //End ic1

    if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS ){
      printf(" -> Returning from FFTlog... \n");
    }

  }
  for(n_thread=0;n_thread<N_threads;++n_thread){
    FFT_planner_free(&(pma->FFT_plan[n_thread]));
    free(integrand1[n_thread]);
    free(integrand2[n_thread]);
  }
  free(integrand1);
  free(integrand2);

  free(pma->FFT_plan);
  *fft_coeff_real_ptr = fft_coeff_real;
  *fft_coeff_imag_ptr = fft_coeff_imag;
  return _SUCCESS_;
}


/**
 * Integrate the Cl's
 *
 * @param ppr                  Input: pointer to precision struct
 * @param pba                  Input: pointer to background struct
 * @param ppt                  Input: pointer to perturbation struct
 * @param pma                  Input: pointer to matter struct
 * @param fft_coeff_real       Input: the fft coefficients (real)
 * @param fft_coeff_imag       Input: the fft coefficients (imaginary)
 * @return the error status
 */
int matter_integrate_cl(struct matters* pma,
                        double * fft_coeff_real,
                        double * fft_coeff_imag){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Integrate Cl's\n");
  }
  /**
   * Initialize and allocate local variables
   * */
  int index_l;
  int index_wd1,index_wd2,index_wd1_wd2;
  int index_cltp1,index_cltp2,index_cltp1_cltp2;

  struct matters_workspace mw;
  struct matters_workspace * pmw = &mw;


  pmw->fft_coeff_real = fft_coeff_real;
  pmw->fft_coeff_imag = fft_coeff_imag;

  /**
   *
   * Now we allocate the fft coefficient arrays
   *  we use to interpolate into
   * We also allocate the bessel integral arrays
   *  we use to interpolate into
   *
   * */

  int tw_max_size = 0;
  if(pma->has_unintegrated_windows){
    tw_max_size = MAX(tw_max_size,pma->tw_size);
  }
  if(pma->has_integrated_windows){
    tw_max_size = MAX(tw_max_size,pma->integrated_tw_size);
  }
  pmw->tau_max_size = tw_max_size;

  /**
   * Now allocate global workspace, and thread workspace
   * */
  class_call(matter_workspace_alloc(pma,pmw),
             pma->error_message,
             pma->error_message);
  class_call(matter_vector_alloc(pma,pmw),
             pma->error_message,
             pma->error_message);


  class_test(pma->uses_integration != matter_integrate_tw_t && pma->uses_integration != matter_integrate_tw_logt,
             pma->error_message,
             "tau integration type not recognized. (Neither tw_t, nor tw_logt) ");



  int abort = _FALSE_;
  #pragma omp parallel private(index_l) firstprivate(pma,pmw)
  {
    if(pma->uses_limber_approximation){
      #pragma omp for
      for(index_l=0;index_l<pma->l_size;++index_l){
        class_call_parallel(matter_get_bessel_limber(pma,
                                                     index_l,
                                                     pmw),
                            pma->error_message,
                            pma->error_message);
      }
      //End l
    }
    else if(pma->has_integrated_windows){
      #pragma omp for
      for(index_l=0;index_l<pma->l_size;++index_l){
        class_call_parallel(matter_get_bessel_fort_parallel_integrated(
                                    pma,
                                    index_l,
                                    pmw),
                            pma->error_message,
                            pma->error_message);
      }
      //End l
    }
    //Ifend limber
  }
  if(abort == _TRUE_) {return _FAILURE_;}


  /**
   * Finally, we allocate the Cl's array
   *  and iterate through all initial conditions and window functions
   * */
  class_alloc(pma->cl,
              pma->cltp_grid_size*sizeof(double*),
              pma->error_message);
  for(index_cltp1=0;index_cltp1<pma->cltp_size;++index_cltp1){
    for(index_cltp2=index_cltp1;index_cltp2<pma->cltp_size;++index_cltp2){
      index_cltp1_cltp2 = index_symmetric_matrix(index_cltp1,index_cltp2,pma->cltp_size);
      {
          {
          class_alloc(pma->cl[index_cltp1_cltp2],
                      pma->window_size[index_cltp1_cltp2]*pma->l_size*sizeof(double),
                      pma->error_message);
          int win_counter = 0;
          for(index_wd1=0;index_wd1<pma->num_windows_per_cltp[index_cltp1];++index_wd1){ //[NS]: modified for DESC
            for(index_wd2=pma->window_index_start[index_cltp1_cltp2][index_wd1];index_wd2<=pma->window_index_end[index_cltp1_cltp2][index_wd1];++index_wd2){
              index_wd1_wd2 = index_symmetric_matrix(index_wd1,index_wd2,pma->num_windows);
              if(pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION){
              //  printf(" -> Calculating for Window Combination %5d,%5d (%10d/%10d) \n",index_wd1,index_wd2,index_wd1_wd2,(pma->num_windows*(pma->num_windows+1))/2)
                printf(" -> Calculating for Window Combination %5d,%5d (%10d/%10d) \n",index_wd1,index_wd2,win_counter,pma->window_size[index_cltp1_cltp2]);
              }



              /**
               * Save information in global workspace
               * */
              pmw->index_wd1 = index_wd1;
              pmw->index_wd2 = index_wd2;
              pmw->index_cltp1 = index_cltp1;
              pmw->index_cltp2 = index_cltp2;
              pmw->index_wd1_wd2 = index_symmetric_matrix(pmw->index_wd1,pmw->index_wd2,pma->num_windows);
              pmw->index_cltp1_cltp2 = index_symmetric_matrix(pmw->index_cltp1,pmw->index_cltp2,pma->cltp_size);
              pmw->window_counter = win_counter;

              /**
               * Now integrate the actual integral
               * */
              class_call(matter_integrate_each(pma,
                                               pmw),
                         pma->error_message,
                         pma->error_message);
              win_counter++;
            }
            //End wd2
          }
          //End wd1
        }
        //End ic2
      }
      //End ic1
    }
    //End cltp2
  }
  //End cltp1
  /**
   * Finally print the obtained results
   * */
  if(pma->matter_verbose > MATTER_VERBOSITY_CLRESULTS){
    /**
     * Print the direct C_l's
     * */
    printf("RESULTS C_l = \n\n");
    printf(" -> l sampling : \n");
    for(index_l=0;index_l<pma->l_size;++index_l){
      printf("%i,",(int)pma->l_sampling[index_l]);
    }
    printf("\n");
    for(index_cltp1=0;index_cltp1<pma->cltp_size;++index_cltp1){
      for(index_cltp2=index_cltp1;index_cltp2<pma->cltp_size;++index_cltp2){
        index_cltp1_cltp2 = index_symmetric_matrix(index_cltp1,index_cltp2,pma->cltp_size);
        printf(" -> At cltp (%4d,%4d) \n",index_cltp1,index_cltp2);
        {
          index_wd1_wd2=0;
          for(index_wd1=0;index_wd1<pma->num_windows_per_cltp[index_cltp1];++index_wd1){
            for(index_wd2=pma->window_index_start[index_cltp1_cltp2][index_wd1];index_wd2<=pma->window_index_end[index_cltp1_cltp2][index_wd1];++index_wd2){
              printf(" -> At win (%4d,%4d) ... \n",index_wd1,index_wd2);
              printf("%.10e",
                pma->cl[index_cltp1_cltp2][index_wd1_wd2*pma->l_size+0]
              );
              for(index_l=1;index_l<pma->l_size;++index_l){
                printf(",%.10e",
                  pma->cl[index_cltp1_cltp2][index_wd1_wd2*pma->l_size+index_l]
                );
              }
              printf("\n");
              //End l
              index_wd1_wd2++;
            }
            //End wd2
          }
          //End wd1
        }
        //End icgrid
      }
      //End cltp1
    }
    //End cltp2
    printf("\n");
    /**
     * Now also print the l(l+1)/2pi C_l's
     * */
    printf("RESULTS l(l+1)/2pi C_l = \n\n");
    printf(" -> l sampling : \n");
    for(index_l=0;index_l<pma->l_size;++index_l){
      printf("%i,", (int)pma->l_sampling[index_l]);
    }
    printf("\n");
    for(index_cltp1=0;index_cltp1<pma->cltp_size;++index_cltp1){
      for(index_cltp2=index_cltp1;index_cltp2<pma->cltp_size;++index_cltp2){
        index_cltp1_cltp2 = index_symmetric_matrix(index_cltp1,index_cltp2,pma->cltp_size);
        printf(" -> At cltp (%4d,%4d) \n",index_cltp1,index_cltp2);
        {
          index_wd1_wd2=0;
          for(index_wd1=0;index_wd1<pma->num_windows_per_cltp[index_cltp1];++index_wd1){
            for(index_wd2=pma->window_index_start[index_cltp1_cltp2][index_wd1];index_wd2<=pma->window_index_end[index_cltp1_cltp2][index_wd1];++index_wd2){
              printf(" -> At win (%4d,%4d) ... \n",index_wd1,index_wd2);
              printf("%.10e",
                pma->l_sampling[0]*(pma->l_sampling[0]+1.0)/(_TWOPI_)*pma->cl[index_cltp1_cltp2][index_wd1_wd2*pma->l_size+0]
              );
              for(index_l=1;index_l<pma->l_size;++index_l){
                printf(",%.10e",
                  pma->l_sampling[index_l]*(pma->l_sampling[index_l]+1.0)/(_TWOPI_)*pma->cl[index_cltp1_cltp2][index_wd1_wd2*pma->l_size+index_l]
                );
              }
              printf("\n");
              //End l
              index_wd1_wd2++;
            }
            //End wd2
          }
          //End wd1
        }
        //End icgrid
      }
      //End cltp1
    }
    //End cltp2
    printf("\n\n");
  }
  //Ifend Cl printing


  /**
   * Finally delete also the temporary arrays
   *  for the coefficients and the bessel functions
   * */
  class_call(matter_vector_free(pma,pmw),
             pma->error_message,
             pma->error_message);
  /**
   * Finally free workspace again
   * */
  class_call(matter_workspace_free(pma,pmw),
             pma->error_message,
             pma->error_message);


  return _SUCCESS_;
}

/**
 * Get the integrand of the cosmological function in t and tau
 * for the the t range where 1>2, and the whole tau range
 *
 * @param pba                  Input: pointer to background struct
 * @param ppt                  Input: pointer to perturbation struct
 * @param pma                  Input: pointer to matter struct
 * @param t                    Input: current value of t
 * @param index_ic1            Input: index of initial condition 1
 * @param index_ic2            Input: index of initial condition 2
 * @param index_radtp1         Input: index of radial type 1
 * @param index_radtp2         Input: index of radial type 2
 * @param index_stp1_stp2      Input: index of source type combo 1,2
 * @param index_wd1            Input: index of window 1
 * @param index_wd2            Input: index of window 2
 * @param integrand_real       Output: the integrand (real part)
 * @param integrand_imag       Output: the integrand (imaginary part)
 * @param wint_fft_real        Input: temporary array for fft coefficients
 * @param wint_fft_imag        Input: temporary array for fft coefficients
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_get_half_integrand(struct matters* pma,
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
                              ){
  int index_ic1_ic2 = pmw->index_ic1_ic2;
  int index_tw_local;
  double* fft_real = pmw->fft_coeff_real;
  double* fft_imag = pmw->fft_coeff_imag;

  double window0_val,window1_val;
  int inf0=0,inf1=0;
  double x0,x1;

  int fft_index00,fft_index01,fft_index10,fft_index11;
  int index_coeff;
  short x1flag;
  double h0,a0,b0;
  double h1,a1,b1;
  double *fft_00_ptr,*fft_01_ptr,*fft_10_ptr,*fft_11_ptr;

  class_call(matter_spline_prepare_hunt(
                                pma->tau_sampling,
                                pma->tau_size,
                                pmw->tau_sampling[index_wd1*pmw->tau_size+0],
                                &inf0,
                                pma->error_message),
            pma->error_message,
            pma->error_message);
  class_call(matter_spline_prepare_hunt(
                                pma->tau_sampling,
                                pma->tau_size,
                                pma->tau0*(1-t)+t*pmw->tau_sampling[index_wd1*pmw->tau_size+0],
                                &inf1,
                                pma->error_message),
            pma->error_message,
            pma->error_message);
  for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){

    if(!pma->uses_separability){
      x1flag = _TRUE_;

      if(pma->uses_intxi_logarithmic && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
        x0 = pma->tau0-pma->exp_integrated_tw_sampling[index_wd1*pmw->tau_size+index_tw_local];//exp(pmw->tau_sampling[index_wd1*pmw->tau_size+index_tw_local]);
        x1 = pma->tau0-t*pma->exp_integrated_tw_sampling[index_wd1*pmw->tau_size+index_tw_local];// - pma->small_log_offset;
      }
      else{
        x0 = pmw->tau_sampling[index_wd1*pmw->tau_size+index_tw_local];
        x1 = pma->tau0-t*(pma->tau0-x0);// - pma->small_log_offset;
      }
      class_test(x0>pma->tau0,
                 pma->error_message,
                 "with x0 = %.10e , t = %.10e ,tau0 = %.10e , x0 = %.10e",x0,t,pma->tau0,x0);
      class_test(x1>pma->tau0,
                 pma->error_message,
                 "with x0 = %.10e , t = %.10e ,tau0 = %.10e , x1 = %.10e",x0,t,pma->tau0,x1);
      if(
        (x1>pma->tw_max[pmw->window_offset+index_wd2] && (!(pma->has_integrated_windows && matter_is_integrated(index_radtp2))))
        ||x1<pma->tw_min[pmw->window_offset+index_wd2]){
        //The point x1 is outside of the window w2
        x1flag=_FALSE_;
      }
      class_call(matter_spline_hunt(pma->tau_sampling,
                                    pma->tau_size,
                                    x0,//285+pma->small_log_offset,
                                    &inf0,
                                    &h0,
                                    &a0,
                                    &b0,
                                    pma->error_message),
                 pma->error_message,
                 pma->error_message);
      if(x1flag==_TRUE_){
       class_call(matter_spline_hunt(pma->tau_sampling,
                                     pma->tau_size,
                                     x1,//285+pma->small_log_offset,
                                     &inf1,
                                     &h1,
                                     &a1,
                                     &b1,
                                     pma->error_message),
                 pma->error_message,
                 pma->error_message);
      }
      if(x1flag==_TRUE_){
        fft_index00 = (inf0)*pma->tau_size+(inf1);
        fft_index01 = (inf0)*pma->tau_size+(inf1+1);
        fft_index10 = (inf0+1)*pma->tau_size+(inf1);
        fft_index11 = (inf0+1)*pma->tau_size+(inf1+1);
        fft_00_ptr = fft_real+fft_index00*pma->size_fft_input;
        fft_01_ptr = fft_real+fft_index01*pma->size_fft_input;
        fft_10_ptr = fft_real+fft_index10*pma->size_fft_input;
        fft_11_ptr = fft_real+fft_index11*pma->size_fft_input;
        for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
          wint_fft_real[index_tw_local][index_coeff] = a0*a1*fft_00_ptr[index_coeff]+a0*b1*fft_01_ptr[index_coeff]+b0*a1*fft_10_ptr[index_coeff]+b0*b1*fft_11_ptr[index_coeff];
        }
        fft_00_ptr = fft_imag+fft_index00*pma->size_fft_input;
        fft_01_ptr = fft_imag+fft_index01*pma->size_fft_input;
        fft_10_ptr = fft_imag+fft_index10*pma->size_fft_input;
        fft_11_ptr = fft_imag+fft_index11*pma->size_fft_input;
        for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
          wint_fft_imag[index_tw_local][index_coeff] = a0*(a1*fft_00_ptr[index_coeff]+b1*fft_01_ptr[index_coeff])+b0*(a1*fft_10_ptr[index_coeff]+b1*fft_11_ptr[index_coeff]);
        }
      }
      else{
        memset(wint_fft_real[index_tw_local],0,pma->size_fft_result*sizeof(double));
        memset(wint_fft_imag[index_tw_local],0,pma->size_fft_result*sizeof(double));
      }
      //End if x1
    }else{
      wint_fft_real[index_tw_local]=fft_real;
      wint_fft_imag[index_tw_local]=fft_imag;
      wint_fft_real[pmw->tau_size+index_tw_local]=fft_real;
      wint_fft_imag[pmw->tau_size+index_tw_local]=fft_imag;
    }
  }
  //End tw
  int last_index0,last_index1;
  class_call(matter_spline_prepare_hunt(
                              pmw->tau_sampling+index_wd1*pmw->tau_size,
                              pmw->tau_size,
                              pmw->tau_sampling[index_wd1*pmw->tau_size],
                              &last_index0,
                              pma->error_message),
            pma->error_message,
            pma->error_message);
  class_call(matter_spline_prepare_hunt(
                              pmw->tau_sampling+index_wd1*pmw->tau_size,
                              pmw->tau_size,
                              pma->tau0*(1-t)+t*pmw->tau_sampling[index_wd1*pmw->tau_size],
                              &last_index1,
                              pma->error_message),
            pma->error_message,
            pma->error_message);

  int derivative_type1 = 0;
  int derivative_type2 = 0;
  class_call(matter_get_derivative_type(pma,
                             &derivative_type1,
                             &derivative_type2,
                             index_radtp1,
                             index_radtp2),
             pma->error_message,
             pma->error_message);
  for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
    if(pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2 && pma->uses_intxi_logarithmic){
      x0 = pma->tau0-pma->exp_integrated_tw_sampling[index_wd1*pmw->tau_size+index_tw_local];//exp(pmw->tau_sampling[index_wd1*pmw->tau_size+index_tw_local]);
    }
    else{
      x0 = pmw->tau_sampling[index_wd1*pmw->tau_size+index_tw_local];
    }
    x1 = pma->tau0-t*(pma->tau0-x0);
    class_call(matter_get_prepared_window_at(pma,
                                             x0,
                                             index_ic1,
                                             index_radtp1,
                                             index_wd1,
                                             &last_index0,
                                             derivative_type1,
                                             &window0_val),
              pma->error_message,
              pma->error_message);
    class_call(matter_get_prepared_window_at(pma,
                                             x1,
                                             index_ic2,
                                             index_radtp2,
                                             index_wd2,
                                             &last_index1,
                                             derivative_type2,
                                             &window1_val),
              pma->error_message,
              pma->error_message);
    double wwval = window0_val*window1_val;
    for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
      integrand_real[index_coeff*pmw->tau_size+index_tw_local] = wwval*wint_fft_real[index_tw_local][index_coeff];
      integrand_imag[index_coeff*pmw->tau_size+index_tw_local] = wwval*wint_fft_imag[index_tw_local][index_coeff];
    }
    //End coeff
  }
  //End tw
  return _SUCCESS_;
}


/**
 * Get the integrand of the cosmological function in t and tau
 * for the whole t and tau ranges
 *
 * @param pba                  Input: pointer to background struct
 * @param ppt                  Input: pointer to perturbation struct
 * @param pma                  Input: pointer to matter struct
 * @param t                    Input: current value of t
 * @param index_ic1            Input: index of initial condition 1
 * @param index_ic2            Input: index of initial condition 2
 * @param index_radtp1         Input: index of radial type 1
 * @param index_radtp2         Input: index of radial type 2
 * @param index_stp1_stp2      Input: index of source type combo 1,2
 * @param index_wd1            Input: index of window 1
 * @param index_wd2            Input: index of window 2
 * @param integrand_real       Output: the integrand (real part)
 * @param integrand_imag       Output: the integrand (imaginary part)
 * @param wint_fft_real        Input: temporary array for fft coefficients
 * @param wint_fft_imag        Input: temporary array for fft coefficients
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_get_ttau_integrand(struct matters* pma,
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
                              ){
  int index_tw_local;
  double* fft_real = pmw->fft_coeff_real;
  double* fft_imag = pmw->fft_coeff_imag;

  double window0_val,window1_val,window2_val;

  int inf0=0,inf1=0,inf2=0;
  double x0,x1,x2;

  double logt = log(t);
  double exp_factor = exp(logt*(pma->nu_real[pmw->index_tilt1_tilt2]-2.0));


  class_test(t==0,
             pma->error_message,
             "stop to avoid division by zero or logarithm of zero");


  int fft_index00,fft_index01,fft_index10,fft_index11;
  int index_coeff;
  short x1flag,x2flag;
  double h0,a0,b0;
  double h1,a1,b1;
  double h2,a2,b2;
  double* cos_val;
  double* sin_val;


  //Theoretically, these could be allocated outside of the t loop,
  //but currently their allocation is not time consuming at all.
  class_alloc(cos_val,
              pma->size_fft_cutoff*sizeof(double),
              pma->error_message);
  class_alloc(sin_val,
              pma->size_fft_cutoff*sizeof(double),
              pma->error_message);
  for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
    double phase = logt*pma->nu_imag[index_coeff];
    cos_val[index_coeff] = cos(phase);
    sin_val[index_coeff] = sin(phase);
  }



  double *fft_00_ptr,*fft_01_ptr,*fft_10_ptr,*fft_11_ptr;
  class_call(matter_spline_prepare_hunt(
                                pma->tau_sampling,
                                pma->tau_size,
                                pmw->tau_sampling[index_wd1*pmw->tau_size+0],
                                &inf0,
                                pma->error_message),
            pma->error_message,
            pma->error_message);
  class_call(matter_spline_prepare_hunt(
                                pma->tau_sampling,
                                pma->tau_size,
                                pma->tau0*(1-t)+t*pmw->tau_sampling[index_wd1*pmw->tau_size+0],
                                &inf1,
                                pma->error_message),
            pma->error_message,
            pma->error_message);
  class_call(matter_spline_prepare_hunt(
                                pma->tau_sampling,
                                pma->tau_size,
                                pma->tau0*(1-1.0/t)+(1.0/t)*pmw->tau_sampling[index_wd1*pmw->tau_size+0],
                                &inf2,
                                pma->error_message),
            pma->error_message,
            pma->error_message);

  for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){

    if(!pma->uses_separability){
      x1flag = _TRUE_;
      x2flag = _TRUE_;

      if(pma->uses_intxi_logarithmic && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
        x0 = pma->tau0-pma->exp_integrated_tw_sampling[index_wd1*pmw->tau_size+index_tw_local];
      }else{
        x0 = pmw->tau_sampling[index_wd1*pmw->tau_size+index_tw_local];
      }
      x1 = pma->tau0-t*(pma->tau0-x0);// - pma->small_log_offset;
      x2 = pma->tau0-(1.0/t)*(pma->tau0-x0);// - pma->small_log_offset;
      class_test(x0>pma->tau0,
                 pma->error_message,
                 "with x0 = %.10e , t = %.10e ,tau0 = %.10e , x0 = %.10e",x0,t,pma->tau0,x0);
      class_test(x1>pma->tau0,
                 pma->error_message,
                 "with x0 = %.10e , t = %.10e ,tau0 = %.10e , x1 = %.10e",x0,t,pma->tau0,x1);
      class_test(x2>pma->tau0,
                 pma->error_message,
                 "with x0 = %.10e , t = %.10e ,tau0 = %.10e , x2 = %.10e",x0,t,pma->tau0,x2);
      if(
        (x1>pma->tw_max[pmw->window_offset+index_wd2] && (!(pma->has_integrated_windows && pmw->is_integrated_radtp2)))
        ||x1<pma->tw_min[pmw->window_offset+index_wd2]){
        //The point x1 is outside of the window w2
        x1flag=_FALSE_;
      }
      if(
        (x2>pma->tw_max[pmw->window_offset+index_wd2] && (!(pma->has_integrated_windows && pmw->is_integrated_radtp2)))
        ||x2<pma->tw_min[pmw->window_offset+index_wd2]){
        //The point x2 is outside of the window w2
        x2flag=_FALSE_;
      }
      class_call(matter_spline_hunt(pma->tau_sampling,
                                    pma->tau_size,
                                    x0,
                                    &inf0,
                                    &h0,
                                    &a0,
                                    &b0,
                                    pma->error_message),
                 pma->error_message,
                 pma->error_message);
      if(x1flag==_TRUE_){
         class_call(matter_spline_hunt(pma->tau_sampling,
                                       pma->tau_size,
                                       x1,
                                       &inf1,
                                       &h1,
                                       &a1,
                                       &b1,
                                       pma->error_message),
                   pma->error_message,
                   pma->error_message);
      }
      if(x2flag==_TRUE_){
          class_call(matter_spline_hunt(pma->tau_sampling,
                                        pma->tau_size,
                                        x2,
                                        &inf2,
                                        &h2,
                                        &a2,
                                        &b2,
                                        pma->error_message),
                     pma->error_message,
                     pma->error_message);
      }

      if(x1flag==_TRUE_){
        fft_index00 = (inf0)*pma->tau_size+(inf1);
        fft_index01 = (inf0)*pma->tau_size+(inf1+1);
        fft_index10 = (inf0+1)*pma->tau_size+(inf1);
        fft_index11 = (inf0+1)*pma->tau_size+(inf1+1);
        fft_00_ptr = fft_real+fft_index00*pma->size_fft_input;
        fft_01_ptr = fft_real+fft_index01*pma->size_fft_input;
        fft_10_ptr = fft_real+fft_index10*pma->size_fft_input;
        fft_11_ptr = fft_real+fft_index11*pma->size_fft_input;
        for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
          wint_fft_real[index_tw_local][index_coeff] = a0*a1*fft_00_ptr[index_coeff]+a0*b1*fft_01_ptr[index_coeff]+b0*a1*fft_10_ptr[index_coeff]+b0*b1*fft_11_ptr[index_coeff];
        }
        fft_00_ptr = fft_imag+fft_index00*pma->size_fft_input;
        fft_01_ptr = fft_imag+fft_index01*pma->size_fft_input;
        fft_10_ptr = fft_imag+fft_index10*pma->size_fft_input;
        fft_11_ptr = fft_imag+fft_index11*pma->size_fft_input;
        for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
          wint_fft_imag[index_tw_local][index_coeff] = a0*(a1*fft_00_ptr[index_coeff]+b1*fft_01_ptr[index_coeff])+b0*(a1*fft_10_ptr[index_coeff]+b1*fft_11_ptr[index_coeff]);
        }
      }
      else{
        memset(wint_fft_real[index_tw_local],0,pma->size_fft_result*sizeof(double));
        memset(wint_fft_imag[index_tw_local],0,pma->size_fft_result*sizeof(double));
      }

      //End if x1
      if(x2flag==_TRUE_){
        fft_index00 = (inf0)*pma->tau_size+(inf2);
        fft_index01 = (inf0)*pma->tau_size+(inf2+1);
        fft_index10 = (inf0+1)*pma->tau_size+(inf2);
        fft_index11 = (inf0+1)*pma->tau_size+(inf2+1);
        fft_00_ptr = fft_real+fft_index00*pma->size_fft_input;
        fft_01_ptr = fft_real+fft_index01*pma->size_fft_input;
        fft_10_ptr = fft_real+fft_index10*pma->size_fft_input;
        fft_11_ptr = fft_real+fft_index11*pma->size_fft_input;
        for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
          wint_fft_real[index_tw_local+pmw->tau_size][index_coeff] = a0*a2*fft_00_ptr[index_coeff]+a0*b2*fft_01_ptr[index_coeff]+b0*a2*fft_10_ptr[index_coeff]+b0*b2*fft_11_ptr[index_coeff];
        }
        fft_00_ptr = fft_imag+fft_index00*pma->size_fft_input;
        fft_01_ptr = fft_imag+fft_index01*pma->size_fft_input;
        fft_10_ptr = fft_imag+fft_index10*pma->size_fft_input;
        fft_11_ptr = fft_imag+fft_index11*pma->size_fft_input;
        for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
          wint_fft_imag[index_tw_local+pmw->tau_size][index_coeff] = a0*(a2*fft_00_ptr[index_coeff]+b2*fft_01_ptr[index_coeff])+b0*(a2*fft_10_ptr[index_coeff]+b2*fft_11_ptr[index_coeff]);
        }

      }
      else{
        memset(wint_fft_real[index_tw_local+pmw->tau_size],0,pma->size_fft_result*sizeof(double));
        memset(wint_fft_imag[index_tw_local+pmw->tau_size],0,pma->size_fft_result*sizeof(double));
      }
      //End if x2
    }else{
      wint_fft_real[index_tw_local]=fft_real;
      wint_fft_imag[index_tw_local]=fft_imag;
      wint_fft_real[pmw->tau_size+index_tw_local]=fft_real;
      wint_fft_imag[pmw->tau_size+index_tw_local]=fft_imag;
    }

  }
  //End tw


  int last_index0,last_index1,last_index2;
  class_call(matter_spline_prepare_hunt(
                              pmw->tau_sampling+index_wd1*pmw->tau_size,
                              pmw->tau_size,
                              pmw->tau_sampling[index_wd1*pmw->tau_size],
                              &last_index0,
                              pma->error_message),
            pma->error_message,
            pma->error_message);
  class_call(matter_spline_prepare_hunt(
                              pmw->tau_sampling+index_wd1*pmw->tau_size,
                              pmw->tau_size,
                              pma->tau0*(1-t)+t*pmw->tau_sampling[index_wd1*pmw->tau_size],
                              &last_index1,
                              pma->error_message),
            pma->error_message,
            pma->error_message);
  class_call(matter_spline_prepare_hunt(
                              pmw->tau_sampling+index_wd1*pmw->tau_size,
                              pmw->tau_size,
                              pma->tau0*(1-1.0/t)+(1.0/t)*pmw->tau_sampling[index_wd1*pmw->tau_size],
                              &last_index2,
                              pma->error_message),
             pma->error_message,
             pma->error_message);


  int derivative_type1 = 0;
  int derivative_type2 = 0;
  class_call(matter_get_derivative_type(pma,
                             &derivative_type1,
                             &derivative_type2,
                             index_radtp1,
                             index_radtp2),
             pma->error_message,
             pma->error_message);

  for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
    if(pma->uses_intxi_logarithmic && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
      x0 = pma->tau0-pma->exp_integrated_tw_sampling[index_wd1*pmw->tau_size+index_tw_local];
    }else{
      x0 = pmw->tau_sampling[index_wd1*pmw->tau_size+index_tw_local];
    }
    x1 = pma->tau0*(1-t)+t*x0;
    x2 = pma->tau0*(1-1.0/t)+(1.0/t)*x0;

    class_call(matter_get_prepared_window_at(pma,
                                             x0,
                                             index_ic1,
                                             index_radtp1,
                                             index_wd1,
                                             &last_index0,
                                             derivative_type1,
                                             &window0_val),
              pma->error_message,
              pma->error_message);
    class_call(matter_get_prepared_window_at(pma,
                                             x1,
                                             index_ic2,
                                             index_radtp2,
                                             index_wd2,
                                             &last_index1,
                                             derivative_type2,
                                             &window1_val),
              pma->error_message,
              pma->error_message);
    class_call(matter_get_prepared_window_at(pma,
                                             x2,
                                             index_ic2,
                                             index_radtp2,
                                             index_wd2,
                                             &last_index2,
                                             derivative_type2,
                                             &window2_val),
              pma->error_message,
              pma->error_message);

    double temp_first,temp_second;
    temp_first = window0_val*window1_val;
    temp_second = exp_factor*window0_val*window2_val;
    for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){//was cutoff?
      integrand_real[index_coeff*pmw->tau_size+index_tw_local] = temp_first*wint_fft_real[index_tw_local][index_coeff]+temp_second*(wint_fft_real[index_tw_local+pmw->tau_size][index_coeff]*cos_val[index_coeff]-wint_fft_imag[index_tw_local+pmw->tau_size][index_coeff]*sin_val[index_coeff]);
      integrand_imag[index_coeff*pmw->tau_size+index_tw_local] = temp_first*wint_fft_imag[index_tw_local][index_coeff]+temp_second*(wint_fft_imag[index_tw_local+pmw->tau_size][index_coeff]*cos_val[index_coeff]+wint_fft_real[index_tw_local+pmw->tau_size][index_coeff]*sin_val[index_coeff]);
    }
    //End coeff
  }
  //End tw
  free(cos_val);
  free(sin_val);
  return _SUCCESS_;
}


/**
 * Small helper function specifying the cosmological function
 * asymptote as a function of t for two windows in non-integrated
 * contributions without derivatives for gaussian windows,
 * to possibly improve interpolation accuracy
 *
 * @param ppr                  Input: pointer to precision struct
 * @param pma                  Input: pointer to matter struct
 * @param t                    Input: current value of t
 * @param index_wd1            Input: current index of window 1
 * @param index_wd2            Input: current index of window 2
 * @param result               Output: pointer to output
 * @return the error status
 */
int matter_asymptote(struct matters* pma,double t, int index_wd1, int index_wd2,double* result){
  double x1 = pma->tau0-0.5*(pma->tw_max[index_wd1]+pma->tw_min[index_wd1]);
  double x2 = pma->tau0-0.5*(pma->tw_max[index_wd2]+pma->tw_min[index_wd2]);

  double sigma1 = 0.5*(pma->tw_max[index_wd1]-pma->tw_min[index_wd1])/pma->selection_cut_at_sigma;
  double sigma2 = 0.5*(pma->tw_max[index_wd2]-pma->tw_min[index_wd2])/pma->selection_cut_at_sigma;
  *result = exp(-0.5*(x1*t-x2)*(x1*t-x2)/(sigma1*sigma1*t*t+sigma2*sigma2))+exp(-0.5*(x2*t-x1)*(x2*t-x1)/(sigma2*sigma2*t*t+sigma1*sigma1));
  return _SUCCESS_;
}


/**
 * Precompute chi^(1-nu) or chi^(2-nu) for normal and logarithmic
 * chi integration respectively
 *
 * dchi = dlog(chi) * chi
 *
 * @param pma                  Input: pointer to matter struct
 * @param index_wd             Input: current window index
 * @param pref_real            Output: array of prefactors (real part)
 * @param pref_imag            Output: array of prefactors (imaginary part)
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_precompute_chit_factors(struct matters* pma,
                                   int index_wd,
                                   double* pref_real,
                                   double* pref_imag,
                                   struct matters_workspace* pmw){
  int is_log = pma->uses_intxi_logarithmic && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2;
  int index_tau;
  double logxi;
  double exp_factor,phase;
  int index_coeff;
  if(is_log == _TRUE_){
    for(index_tau=0;index_tau<pmw->tau_size;++index_tau){
      logxi = pmw->tau_sampling[index_wd*pmw->tau_size+index_tau];
      exp_factor = exp(logxi*(2.0-pma->nu_real[pmw->index_tilt1_tilt2]));
      //exp_factor = exp(logxi*(2.0-pma->bias));
      for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
        phase = -logxi*pma->nu_imag[index_coeff];
        pref_real[index_tau*pma->size_fft_result+index_coeff] = exp_factor*cos(phase);
        pref_imag[index_tau*pma->size_fft_result+index_coeff] = exp_factor*sin(phase);
      }
    }
    //End tw local
  }else{
    for(index_tau=0;index_tau<pmw->tau_size;++index_tau){
      logxi = log((pma->tau0-pmw->tau_sampling[index_wd*pmw->tau_size+index_tau]));//285+pma->small_log_offset));
      exp_factor = exp(logxi*(1.0-pma->nu_real[pmw->index_tilt1_tilt2]));//was 1.0 instead of offset, revised to 1.0 on 23.10
      //exp_factor = exp(logxi*(1.0-pma->bias));
      for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
        phase = -logxi*pma->nu_imag[index_coeff];
        pref_real[index_tau*pma->size_fft_result+index_coeff] = exp_factor*cos(phase);
        pref_imag[index_tau*pma->size_fft_result+index_coeff] = exp_factor*sin(phase);
      }
    }
    //End tw local
  }
  return _SUCCESS_;
}


/**
 * Integrate the cosmological function f_n^ij(t)
 *
 * @param ppr                  Input: pointer to precision struct
 * @param pba                  Input: pointer to background struct
 * @param ppt                  Input: pointer to perturbation struct
 * @param pma                  Input: pointer to matter struct
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_integrate_cosmo(struct matters* pma,
                           struct matters_workspace* pmw
                          ){
  int index_radtp1 = pmw->index_radtp1;
  int index_radtp2 = pmw->index_radtp2;
  int tw_max_size = pmw->tau_max_size;


  double t_min = pmw->t_min;
  double t_max = pmw->t_max;

  int index_spl;
  int index_t,index_coeff,index_tw_local;
  double intxi_local_real,intxi_local_imag;

  int t_size_local = (pma->uses_intxi_interpolation?pma->t_spline_size:pma->t_size);
  short integrate_logarithmically = (pma->uses_integration == matter_integrate_tw_logt);
  double t;

  double* int_real;
  double* int_imag;
  double** window_fft_real;
  double** window_fft_imag;

  double y_min,y_max;
  y_min = -log(1-t_min);
  y_max = -log(1-t_max);

  class_call(matter_precompute_chit_factors(pma,
                                            pmw->index_wd1,
                                            pmw->pref_real,
                                            pmw->pref_imag,
                                            pmw),
             pma->error_message,
             pma->error_message);
  if(pma->uses_intxi_symmetrized && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
    class_call(matter_precompute_chit_factors(pma,
                                              pmw->index_wd2,
                                              pmw->pref_real+pmw->tau_max_size*pma->size_fft_result,
                                              pmw->pref_imag+pmw->tau_max_size*pma->size_fft_result,
                                              pmw),
             pma->error_message,
             pma->error_message);
  }

  int abort = _FALSE_;
  #pragma omp parallel private(index_t,index_coeff,index_tw_local,t,int_real,int_imag,window_fft_real,window_fft_imag) firstprivate(pma,pmw,t_size_local)
  {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif

    int_real = pmw->pmv[tid]->integrand_real;
    int_imag = pmw->pmv[tid]->integrand_imag;
    window_fft_real = pmw->pmv[tid]->window_fft_real;
    window_fft_imag = pmw->pmv[tid]->window_fft_imag;

  /**
   * Now obtain the f_n^{ij}(t) function
   * */
  if(pma->uses_intxi_logarithmic && pma->uses_intxi_symmetrized && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
    /**
     * Now obtain the f_n^{ij}(t) function
     * */
    #pragma omp for
    for(index_t=0;index_t<t_size_local;++index_t){
      matter_get_t(index_t)
      class_call_parallel(matter_get_half_integrand(pma,
                                t,
                                pmw->index_ic1,
                                pmw->index_ic2,
                                index_radtp1,
                                index_radtp2,
                                pmw->index_stp1_stp2,
                                pmw->index_wd1,
                                pmw->index_wd2,
                                int_real,
                                int_imag,
                                window_fft_real,
                                window_fft_imag,
                                pmw),
                  pma->error_message,
                  pma->error_message);
      class_call_parallel(matter_get_half_integrand(pma,
                                t,
                                pmw->index_ic2,
                                pmw->index_ic1,
                                index_radtp2,
                                index_radtp1,
                                pmw->index_stp2_stp1,
                                pmw->index_wd2,
                                pmw->index_wd1,
                                int_real+tw_max_size*pma->size_fft_result,
                                int_imag+tw_max_size*pma->size_fft_result,
                                window_fft_real,
                                window_fft_imag,
                                pmw),
                  pma->error_message,
                  pma->error_message);
      for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
        double sum_real =0.0;
        double sum_imag =0.0;
        for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
          sum_real+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]-pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]);
          sum_imag+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]+pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]);
        }
        for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
          sum_real+=pmw->tau_weights[pmw->index_wd2*pmw->tau_size+index_tw_local]*(pmw->pref_real[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]-pmw->pref_imag[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]);
          sum_imag+=pmw->tau_weights[pmw->index_wd2*pmw->tau_size+index_tw_local]*(pmw->pref_real[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]+pmw->pref_imag[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]);
        }
        //End tw integration
        if(pma->uses_intxi_interpolation){
          pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_real;
          pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_imag;
        }
        else{
          pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_real;
          pmw->intxi_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_imag;
        }
        //Ifend isinterpolated
      }
      //End coeff
    }
    //End t
  }
  else if(pma->uses_intxi_logarithmic && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
    #pragma omp for
    for(index_t=0;index_t<t_size_local;++index_t){
      matter_get_t(index_t)
      class_call_parallel(matter_get_ttau_integrand(pma,
                                t,
                                pmw->index_ic1,
                                pmw->index_ic2,
                                index_radtp1,
                                index_radtp2,
                                pmw->index_stp1_stp2,
                                pmw->index_wd1,
                                pmw->index_wd2,
                                int_real,
                                int_imag,
                                window_fft_real,
                                window_fft_imag,
                                pmw),
                  pma->error_message,
                  pma->error_message);
      for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
        double sum_real =0.0;
        double sum_imag =0.0;
        for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
          sum_real+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]-pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]);
          sum_imag+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]+pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]);
        }
        //End tw integration
        if(pma->uses_intxi_interpolation){
          pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_real;
          pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_imag;
        }
        else{
          pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_real;
          pmw->intxi_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_imag;
        }
        //Ifend isinterpolated
      }
      //End coeff
    }
    //End t
  }
  else if(pma->uses_intxi_symmetrized && pmw->is_integrated_radtp1 && pmw->is_integrated_radtp2){
    /**
     * Now obtain the f_n^{ij}(t) function
     * */
    #pragma omp for
    for(index_t=0;index_t<t_size_local;++index_t){
      matter_get_t(index_t)
      class_call_parallel(matter_get_half_integrand(pma,
                                t,
                                pmw->index_ic1,
                                pmw->index_ic2,
                                index_radtp1,
                                index_radtp2,
                                pmw->index_stp1_stp2,
                                pmw->index_wd1,
                                pmw->index_wd2,
                                int_real,
                                int_imag,
                                window_fft_real,
                                window_fft_imag,
                                pmw),
                  pma->error_message,
                  pma->error_message);
      class_call_parallel(matter_get_half_integrand(pma,
                                t,
                                pmw->index_ic2,
                                pmw->index_ic1,
                                index_radtp2,
                                index_radtp1,
                                pmw->index_stp2_stp1,
                                pmw->index_wd2,
                                pmw->index_wd1,
                                int_real+tw_max_size*pma->size_fft_result,
                                int_imag+tw_max_size*pma->size_fft_result,
                                window_fft_real,
                                window_fft_imag,
                                pmw),
                  pma->error_message,
                  pma->error_message);
      for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
        double sum_real =0.0;
        double sum_imag =0.0;
        for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
          sum_real+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]-pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]);
          sum_imag+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]+pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]);
        }
        for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
          sum_real+=pmw->tau_weights[pmw->index_wd2*pmw->tau_size+index_tw_local]*(pmw->pref_real[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]-pmw->pref_imag[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]);
          sum_imag+=pmw->tau_weights[pmw->index_wd2*pmw->tau_size+index_tw_local]*(pmw->pref_real[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]+pmw->pref_imag[tw_max_size*pma->size_fft_result+index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local+tw_max_size*pma->size_fft_result]);
        }
        //End tw integration
        if(pma->uses_intxi_interpolation){
          pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_real;
          pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_imag;
        }
        else{
          pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_real;
          pmw->intxi_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_imag;

        }
        //Ifend isinterpolated
      }
      //End coeff
    }
    //End t
  }
  else{
    #pragma omp for
    for(index_t=0;index_t<t_size_local;++index_t){
      matter_get_t(index_t)
      class_call_parallel(matter_get_ttau_integrand(pma,
                                t,
                                pmw->index_ic1,
                                pmw->index_ic2,
                                index_radtp1,
                                index_radtp2,
                                pmw->index_stp1_stp2,
                                pmw->index_wd1,
                                pmw->index_wd2,
                                int_real,
                                int_imag,
                                window_fft_real,
                                window_fft_imag,
                                pmw),
                  pma->error_message,
                  pma->error_message);
      for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
        double sum_real =0.0;
        double sum_imag =0.0;
        for(index_tw_local=0;index_tw_local<pmw->tau_size;++index_tw_local){
          sum_real+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]-pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]);
          sum_imag+=pmw->tau_weights[pmw->index_wd1*pmw->tau_size+index_tw_local]*(pmw->pref_real[index_tw_local*pma->size_fft_result+index_coeff]*int_imag[index_coeff*pmw->tau_size+index_tw_local]+pmw->pref_imag[index_tw_local*pma->size_fft_result+index_coeff]*int_real[index_coeff*pmw->tau_size+index_tw_local]);
        }

        //End tw integration
        if(pma->uses_intxi_interpolation){
          pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_real;
          pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] = sum_imag;
        }
        else{
          pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_real;
          pmw->intxi_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = sum_imag;
        }
        //Ifend isinterpolated
      }
      //End coeff
    }
    //End t
  }
  //Ifend
  }
  if(abort == _TRUE_){return _FAILURE_;}
  //End parallel
  if(pma->uses_intxi_interpolation){
    if(pma->uses_intxi_asymptotic && !(pmw->is_integrated_radtp1 || pmw->is_integrated_radtp2)){
      for(index_t=0;index_t<t_size_local;++index_t){
        matter_get_t(index_t)
        double temp;
        class_call(matter_asymptote(pma, t, pmw->index_wd1, pmw->index_wd2, &temp),pma->error_message,pma->error_message);
        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
          pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_t] /= temp;
        }
        //End coeff
      }
      //End t
    }
    //Ifend asymptotic
    /**
     * If we want spline interpolation,
     *  we first have to calculate the splines,
     *  and then we interpolate said splines
     * */
    for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
      class_call(array_spline_table_columns(pma->t_spline_sampling,
                                            pma->t_spline_size,
                                            pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2]+index_coeff*pma->t_spline_size,
                                            1,
                                            pmw->ddintxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2]+index_coeff*pma->t_spline_size,
                                            _SPLINE_EST_DERIV_,
                                            pma->error_message),
           pma->error_message,
           pma->error_message);
      class_call(array_spline_table_columns(pma->t_spline_sampling,
                                            pma->t_spline_size,
                                            pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2]+index_coeff*pma->t_spline_size,
                                            1,
                                            pmw->ddintxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2]+index_coeff*pma->t_spline_size,
                                            _SPLINE_EST_DERIV_,
                                            pma->error_message),
            pma->error_message,
            pma->error_message);
      index_spl = 0;
      for(index_t=0;index_t<pma->t_size;++index_t){
        matter_get_t_orig(index_t);
        double a,b,h;
        class_call(matter_spline_hunt(pma->t_spline_sampling,
                           pma->t_spline_size,
                           pma->t_sampling[index_t],
                           &index_spl,
                           &h,
                           &a,
                           &b,
                           pma->error_message),
                  pma->error_message,
                  pma->error_message);
        intxi_local_real = b*pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl+1]
                          +a*pmw->intxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl]
                    +(
                      (b*b*b-b)*pmw->ddintxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl+1]+
                      (a*a*a-a)*pmw->ddintxi_spline_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl]
                     )*h*h/6.;
        intxi_local_imag = b*pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl+1]
                          +a*pmw->intxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl]
                    +(
                      (b*b*b-b)*pmw->ddintxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl+1]+
                      (a*a*a-a)*pmw->ddintxi_spline_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_spline_size+index_spl]
                     )*h*h/6.;
        pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = intxi_local_real;
        pmw->intxi_imag[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] = intxi_local_imag;
        if(pma->uses_intxi_asymptotic && !(pmw->is_integrated_radtp1 || pmw->is_integrated_radtp2)){
          double temp;
          //TODO :: can be optimized (no coeff dependence)
          class_call(matter_asymptote(pma, t, pmw->index_wd1, pmw->index_wd2, &temp),pma->error_message,pma->error_message);
          pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2][index_coeff*pma->t_size+index_t] *= temp;
        }
        //Ifend asymptotic
      }
      //End t
    }
    //End coeff
  }
  //Ifend interpolation
  return _SUCCESS_;
}


/**
 * Integrate each bin combination of Cl's
 *
 * @param ppr                  Input: pointer to precision struct
 * @param pba                  Input: pointer to background struct
 * @param ppt                  Input: pointer to perturbation struct
 * @param pma                  Input: pointer to matter struct
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_integrate_each(struct matters* pma,
                          struct matters_workspace * pmw
                          ){
  /**
   * Define and allocate local variables
   * */
  int index_coeff;
  int index_radtp1,index_radtp2;
  int index_radtp_of_bitp1,index_radtp_of_bitp2;
  int index_bitp1,index_bitp2;
  int index_tilt1,index_tilt2,index_tilt1_tilt2;

  double sum_temp = 0.0;
  int index_l;
  double t_min,t_max,y_min,y_max;

  int index_t;
  double t;
  double intxi_local_real,intxi_local_imag;
  int integrated_t_offset;
  short integrate_logarithmically = (pma->uses_integration == matter_integrate_tw_logt);
  int print_total_index = 0;



  double* sum_l;
  class_alloc(sum_l,
              pma->l_size*sizeof(double),
              pma->error_message);
  memset(sum_l,0,pma->l_size*sizeof(double));



  /**
   * Get the necessary bessel integrals
   *   and store in pre-prepared arrays
   * */
  if(pma->has_unintegrated_windows){
    int abort = _FALSE_;
    #pragma omp parallel private(index_l) firstprivate(pma,pmw)
    {
      #pragma omp for
      for(index_l=0;index_l<pma->l_size;++index_l){
        if(!pma->uses_limber_approximation){
          class_call_parallel(matter_get_bessel_fort_parallel(pma,
                          index_l,
                          pmw
                          ),
                  pma->error_message,
                  pma->error_message);
        }
        //Ifend limber
      }
      //End l
    }
    if(abort == _TRUE_) {return _FAILURE_;}
  }

  /**
   * Now iterate through all bessel integral types
   *  and radial types
   * */
  short switched_flag = _FALSE_;
  short type_doubling = _FALSE_;
  for(index_bitp1 = 0; index_bitp1< pma->bitp_size;++index_bitp1){
    if(pma->has_bitp_normal && index_bitp1 == pma->bitp_index_normal){
      index_tilt1 = pma->tilt_index_normal;
    }
    else{
      index_tilt1 = pma->tilt_index_reduced;
    }
    for(index_bitp2 = 0; index_bitp2< pma->bitp_size;++index_bitp2){
      if(pma->has_bitp_normal && index_bitp2 == pma->bitp_index_normal){
        index_tilt2 = pma->tilt_index_normal;
      }
      else{
        index_tilt2 = pma->tilt_index_reduced;
      }
      index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);
      pmw->index_tilt1_tilt2 = index_tilt1_tilt2;

      for(index_radtp_of_bitp1 =0; index_radtp_of_bitp1 < pma->radtp_of_bitp_size[pmw->index_cltp1*pma->bitp_size+index_bitp1];++index_radtp_of_bitp1){
        for(index_radtp_of_bitp2 =0; index_radtp_of_bitp2 < pma->radtp_of_bitp_size[pmw->index_cltp2*pma->bitp_size+index_bitp2]; ++index_radtp_of_bitp2){
          index_radtp1 = pma->radtps_of_bitp[pmw->index_cltp1*pma->bitp_size+index_bitp1][index_radtp_of_bitp1];
          index_radtp2 = pma->radtps_of_bitp[pmw->index_cltp2*pma->bitp_size+index_bitp2][index_radtp_of_bitp2];
          pmw->index_radtp1 = index_radtp1;
          pmw->index_radtp2 = index_radtp2;
          /* Do test of skipping over a type due to symmetry in all other indices */
          if((pmw->index_wd1==pmw->index_wd2) && (pmw->index_cltp1 == pmw->index_cltp2)){
            if(index_radtp2 > index_radtp1){type_doubling = _FALSE_;continue;}
            else if(index_radtp2 < index_radtp1){type_doubling = _TRUE_;}
            else{type_doubling = _FALSE_;}
          }else{type_doubling = _FALSE_;}
          /* Check for swapping around types to reduce computational effort of cross terms */
          if(matter_is_integrated(index_radtp1) && !(matter_is_integrated(index_radtp2))){
            class_call(matter_swap_workspace(pmw),
                       pma->error_message,
                       pma->error_message);
            switched_flag = _TRUE_;
          }

          index_radtp1 = pmw->index_radtp1;
          index_radtp2 = pmw->index_radtp2;
          pmw->is_integrated_radtp1 = matter_is_integrated(index_radtp1);
          pmw->is_integrated_radtp2 = matter_is_integrated(index_radtp2);

          if(pma->uses_density_splitting && pma->has_stp_delta_m && pma->uses_limber_approximation && (index_radtp1 == pma->radtp_dens1 || index_radtp2 == pma->radtp_dens1)){
            continue;
          }

          print_total_index++;
          if(pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION && !MATTER_REWRITE_PRINTING){
            printf(" -> BI types [%1d,%1d] (sizes [%2d,%2d]), RAD types [%2d,%2d] (Total %3d/%3d)",index_bitp1,index_bitp2,pma->radtp_of_bitp_size[index_bitp1],pma->radtp_of_bitp_size[index_bitp2],index_radtp1,index_radtp2,print_total_index,pma->radtp_size_total*pma->radtp_size_total);
          }
          if(pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION && MATTER_REWRITE_PRINTING){
            printf("\r -> BI types [%1d,%1d] (sizes [%2d,%2d]), RAD types [%2d,%2d] (Total %3d/%3d)",index_bitp1,index_bitp2,pma->radtp_of_bitp_size[index_bitp1],pma->radtp_of_bitp_size[index_bitp2],index_radtp1,index_radtp2,print_total_index,pma->radtp_size_total*pma->radtp_size_total);
            fflush(stdout);
          }

          /**
           * First define correct t sampling
           * */
          matter_get_t_limits(pmw->index_wd1,pmw->index_wd2)
          pmw->tau_weights = pma->tw_weights;
          pmw->tau_size = pma->tw_size;
          pmw->tau_sampling = pma->tw_sampling;
          if(pmw->is_integrated_radtp1){
            t_min = 0.0+pma->bi_maximal_t_offset;//186+pma->bi_maximal_t_offset;
            t_max = 1.0-pma->bi_maximal_t_offset;//186-pma->bi_maximal_t_offset found important;
            pmw->tau_weights = pma->integrated_tw_weights;
            pmw->tau_size = pma->integrated_tw_size;
            pmw->tau_sampling = pma->integrated_tw_sampling;
          }
          pmw->window_offset = 0;
          if(pmw->is_integrated_radtp2){
            t_min = 0.0+pma->bi_maximal_t_offset;//186+pma->bi_maximal_t_offset;
            t_max = 1.0-pma->bi_maximal_t_offset;//186-pma->bi_maximal_t_offset found important;
            pmw->window_offset = pma->num_windows_per_cltp[pmw->index_cltp1];
          }

          if(pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION && pma->matter_verbose > MATTER_VERBOSITY_RANGES && !MATTER_REWRITE_PRINTING){
            printf(" -> t range from %.10e to %.10e \n",t_min,t_max);
          }

          //TODO :: reformulate
          class_test(t_min >= t_max,
                     pma->error_message,
                     "Adjust matter_t_offset \n");
          y_min = -log(1-t_min);
          y_max = -log(1-t_max);
          integrated_t_offset = pma->t_size*((pmw->is_integrated_radtp1 || pmw->is_integrated_radtp2)?1:0);//pma->t_size*(is_integrated1*2+is_integrated2);

          pmw->t_min = t_min;
          pmw->t_max = t_max;

          class_call(matter_integrate_cosmo(pma,
                                            pmw),
                       pma->error_message,
                       pma->error_message);
          /**
           * We have now obtained the f_n^{ij}(t)
           *
           * Now we can advance to integrating the final Cl's
           * */
          //#pragma OMP parallel for firstprivate(pma,pmw,index_radtp2,index_radtp1,index_tilt1_tilt2,integrated_t_offset,t_max,t_min,y_max,y_min,sum_l) private(index_t, index_l, index_coeff,sum_temp, t,intxi_local_real,intxi_local_imag)
          double* ixr = pmw->intxi_real[index_radtp1*pma->radtp_size_total+index_radtp2];
          double* ixi = pmw->intxi_imag[index_radtp1*pma->radtp_size_total+index_radtp2];
          for(index_l=0;index_l<pma->l_size;++index_l){
            double** wbr = pmw->window_bessel_real[index_l];
            double** wbi = pmw->window_bessel_imag[index_l];
            double sum_t = 0.0;
            for(index_t=0;index_t<pma->t_size;++index_t){
              matter_get_t_orig(index_t)
              sum_temp = 0.0;
              double bes_local_real,bes_local_imag;

              /**
               * The reverse FFT has
               *
               *  index 0 with a factor of 1
               *  index N with a factor of 1
               *  index i with a factor of 2,
               *   since here there would theoretically be
               *   both index i and N-i,
               *   but we relate N-i to i by symmetry
               *   of having a real integrand
               *
               * */

              intxi_local_real = ixr[0*pma->t_size+index_t];
              intxi_local_imag = ixi[0*pma->t_size+index_t];
              bes_local_real = wbr[index_tilt1_tilt2*pma->size_fft_result+0][index_t+integrated_t_offset];
              bes_local_imag = wbi[index_tilt1_tilt2*pma->size_fft_result+0][index_t+integrated_t_offset];
              sum_temp +=intxi_local_real*bes_local_real-intxi_local_imag*bes_local_imag;

              if(pma->size_fft_cutoff==pma->size_fft_result){
                intxi_local_real = ixr[(pma->size_fft_result-1)*pma->t_size+index_t];
                intxi_local_imag = ixi[(pma->size_fft_result-1)*pma->t_size+index_t];
                bes_local_real = wbr[index_tilt1_tilt2*pma->size_fft_result+(pma->size_fft_result-1)][index_t+integrated_t_offset];
                bes_local_imag = wbi[index_tilt1_tilt2*pma->size_fft_result+(pma->size_fft_result-1)][index_t+integrated_t_offset];
                sum_temp +=intxi_local_real*bes_local_real-intxi_local_imag*bes_local_imag;
              }
              for(index_coeff=1;index_coeff<pma->size_fft_cutoff-1;++index_coeff){
                intxi_local_real = ixr[index_coeff*pma->t_size+index_t];
                intxi_local_imag = ixi[index_coeff*pma->t_size+index_t];
                bes_local_real = wbr[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t+integrated_t_offset];
                bes_local_imag = wbi[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t+integrated_t_offset];
                sum_temp +=2.0*(intxi_local_real*bes_local_real-intxi_local_imag*bes_local_imag);
              }
              if((pma->uses_integration == matter_integrate_tw_logt) && !pma->uses_limber_approximation){
                sum_t+=(1-t)*pma->t_weights[index_t]*sum_temp;
              }
              else{
                sum_t+=pma->t_weights[index_t]*sum_temp;
              }
            }
            /**
             * Since we only integrate over t_min to t_max,
             *  we have to rescale the final value obtained
             * (The t_weights are always scaled as 0 to 1)
             * */
            if(!(pma->uses_integration == matter_integrate_tw_logt) && !pma->uses_limber_approximation){
              sum_t*=(t_max-t_min);
            }
            else if(!pma->uses_limber_approximation){
              sum_t*=(y_max-y_min);
            }
            if(pma->has_bitp_lfactor && index_bitp2 == pma->bitp_index_lfactor){
              sum_t*=sqrt((pma->l_sampling[index_l]-1.0)*pma->l_sampling[index_l]*(pma->l_sampling[index_l]+1.0)*(pma->l_sampling[index_l]+2.0));
            }
            if(pma->has_bitp_lfactor && index_bitp1 == pma->bitp_index_lfactor){
              sum_t*=sqrt((pma->l_sampling[index_l]-1.0)*pma->l_sampling[index_l]*(pma->l_sampling[index_l]+1.0)*(pma->l_sampling[index_l]+2.0));
            }
            if(type_doubling == _TRUE_){sum_t*=2.;}
            sum_l[index_l] += sum_t;
            if(pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION_PARTIAL && !MATTER_REWRITE_PRINTING){
              printf("(l:%i (bitp1 =%i ,bitp2 =%i, radtp1 =%i , radtp2 = %i) = %.10e (total = %.10e) \n",(int)pma->l_sampling[index_l],index_bitp1,index_bitp2,index_radtp1,index_radtp2,sum_t,sum_l[index_l]);
            }
          }

          //End l
          if(switched_flag==_TRUE_){
            class_call(matter_swap_workspace(pmw),
                       pma->error_message,
                       pma->error_message);
            switched_flag = _FALSE_;
          }

          index_radtp1 = pmw->index_radtp1;
          index_radtp2 = pmw->index_radtp2;
        }
        //End radtp2
      }
      //End radtp1
    }
    //End bitp2
  }
  //End bitp1


  /**
   * Print final results
   * */
  if(MATTER_REWRITE_PRINTING && pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION){
    printf("\r -> Output for current Window Combination :                                 \n");
  }
  for(index_l=0;index_l<pma->l_size;++index_l){
    if(pma->matter_verbose > MATTER_VERBOSITY_CLCALCULATION){
      printf("(l:%i) = %.10e \n",(int)pma->l_sampling[index_l],sum_l[index_l]);
    }
    pma->cl[pmw->index_cltp1_cltp2][pmw->window_counter*pma->l_size+index_l] = sum_l[index_l];
  }


  free(sum_l);
  return _SUCCESS_;
}


/**
 * Small helper function to get bessel integrals for every value of t
 * in the limber approximation
 *
 * @param pma                  Input: pointer to matter struct
 * @param index_l              Input: l index
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_get_bessel_limber(
                            struct matters* pma,
                            int index_l,
                            struct matters_workspace * pmw
                            ){
  double** window_bessel_real = pmw->window_bessel_real[index_l];
  double** window_bessel_imag = pmw->window_bessel_imag[index_l];
  int index_tilt1_tilt2;
  int index_coeff;
  double l0 = sqrt(pma->l_sampling[index_l]*(pma->l_sampling[index_l]+1.));
  for(index_tilt1_tilt2=0;index_tilt1_tilt2<pma->tilt_grid_size;++index_tilt1_tilt2){
    for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
      double exp_factor = exp(log(l0)*(pma->nu_real[index_tilt1_tilt2]-3.));
      double phase = log(l0)*pma->nu_imag[index_coeff];
      /**
       * Since at t=1, we cannot split integration into 1/t and t,
       *  only half of the single point t=1 contributes
       * The theoretical factor would be (2pi^2),
       *  but we get aformentioned additional factor of 1/2.
       * */
      window_bessel_real[index_tilt1_tilt2*pma->size_fft_result+index_coeff][0] = _PI_*_PI_*exp_factor*cos(phase);
      window_bessel_imag[index_tilt1_tilt2*pma->size_fft_result+index_coeff][0] = _PI_*_PI_*exp_factor*sin(phase);
      if(pma->has_integrated_windows){
        window_bessel_real[index_tilt1_tilt2*pma->size_fft_result+index_coeff][0+pma->t_size] = _PI_*_PI_*exp_factor*cos(phase);
        window_bessel_imag[index_tilt1_tilt2*pma->size_fft_result+index_coeff][0+pma->t_size] = _PI_*_PI_*exp_factor*sin(phase);
      }
    }
    //End coeff
  }
  //End tilt grid
  return _SUCCESS_;
}


/**
 * Small helper function to get the type of derivative acting
 * on the window function depending on the radial type
 *
 * @param pma                  Input: pointer to matter struct
 * @param derivative_type1     Output: pointer to first derivative type
 * @param derivative_type2     Output: pointer to second derivative type
 * @param index_radtp1         Input: radial type of first source
 * @param index_radtp2         Input: radial type of second source
 * @return the error status
 */
int matter_get_derivative_type(
                               struct matters* pma,
                               int* derivative_type1,
                               int* derivative_type2,
                               int index_radtp1,
                               int index_radtp2
                              ){
  if(!pma->uses_relative_factors){
    if(pma->has_stp_delta_m && pma->uses_density_splitting){
      if(index_radtp1 == pma->radtp_dens1){
        *derivative_type1 = 4;
      }
      if(index_radtp2 == pma->radtp_dens1){
        *derivative_type2 = 4;
      }
    }
    if(pma->has_redshift_space_distortion && (!pma->uses_rsd_combination)){
      if(index_radtp1 == pma->radtp_dop1){
        *derivative_type1 = 1;
      }
      if(index_radtp1 == pma->radtp_rsd){
        *derivative_type1 = 2;
      }
      if(index_radtp2 == pma->radtp_dop1){
        *derivative_type2 = 1;
      }
      else if(index_radtp2 == pma->radtp_rsd){
        *derivative_type2 = 2;
      }
    }
  }
  if(pma->has_lensing_terms){
    if(index_radtp1 == pma->radtp_nclens){
      *derivative_type1 = -1;
    }
    if(index_radtp2 == pma->radtp_nclens){
      *derivative_type2 = -1;
    }
  }
  if(pma->has_cl_shear){
    if(index_radtp1 == pma->radtp_shlens){
      *derivative_type1 = -1;
    }
    if(index_radtp2 == pma->radtp_shlens){
      *derivative_type2 = -1;
    }
  }
  if(pma->has_gravitational_terms){
    if(index_radtp1 == pma->radtp_g4 || index_radtp1 == pma->radtp_g5){
      *derivative_type1 = -1;
    }
    if(index_radtp2 == pma->radtp_g4 || index_radtp2 == pma->radtp_g5){
      *derivative_type2 = -1;
    }
  }
  return _SUCCESS_;
}


/**
 * Small helper function to get bessel integrals for every value of t
 * in a parrallelized code for non-integrated contributions
 *
 * @param pba                  Input: pointer to background struct
 * @param pma                  Input: pointer to matter struct
 * @param index_l              Input: l index
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_get_bessel_fort_parallel(
                      struct matters* pma,
                      int index_l,
                      struct matters_workspace* pmw
                      ){
  int index_l_eval = (pma->uses_bessel_storeall?pma->l_sampling[index_l]:index_l);
  int index_wd1 = pmw->index_wd1;
  int index_wd2 = pmw->index_wd2;
  double** window_bessel_real = pmw->window_bessel_real[index_l];
  double** window_bessel_imag = pmw->window_bessel_imag[index_l];
  short integrate_logarithmically = (pma->uses_integration == matter_integrate_tw_logt);
  int index_coeff;
  double res_real,res_imag;
  int last_t = 0;
  double t_min,t_max;
  int index_tilt1,index_tilt2,index_tilt1_tilt2;
  matter_get_t_limits(index_wd1,index_wd2)
  double y_min,y_max;
  y_min = -log(1-t_min);
  y_max = -log(1-t_max);
  int index_t;
  double t,eval_pt=0.0;
  double a,b,h;
  double* bi_real_i;
  double* bi_imag_i;
  double* ddbi_real_i;
  double* ddbi_imag_i;
  if(pma->has_unintegrated_windows){
    for(index_tilt1=0;index_tilt1<pma->tilt_size;++index_tilt1){
      for(index_tilt2=index_tilt1;index_tilt2<pma->tilt_size;++index_tilt2){
        index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);

        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
          last_t = pma->bi_size[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff]-2;
          bi_real_i = pma->bi_real[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          bi_imag_i = pma->bi_imag[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          ddbi_real_i = pma->ddbi_real[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          ddbi_imag_i = pma->ddbi_imag[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          // HERE :: last_t t change
          last_t = 0;
          for(index_t=0;index_t<pma->t_size;++index_t){
            matter_get_t_orig(index_t);
            eval_pt = 1.0-t;
            if(eval_pt>pma->bi_max[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff]){
              window_bessel_real[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t] = 0.0;
              window_bessel_imag[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t] = 0.0;
              continue;
            }
            class_call(matter_spline_hunt(pma->bi_sampling,
                                          pma->bi_size[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff],
                                          eval_pt,
                                          &last_t,
                                          &h,
                                          &a,
                                          &b,
                                          pma->error_message),
                         pma->error_message,
                         pma->error_message);
            res_real =a * bi_real_i[last_t] + b * bi_real_i[last_t+1] + ((a*a*a-a)* ddbi_real_i[last_t] + (b*b*b-b)* ddbi_real_i[last_t+1])*h*h/6.;
            res_imag =a * bi_imag_i[last_t] + b * bi_imag_i[last_t+1] + ((a*a*a-a)* ddbi_imag_i[last_t] + (b*b*b-b)* ddbi_imag_i[last_t+1])*h*h/6.;
            window_bessel_real[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t] = res_real;
            window_bessel_imag[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t] = res_imag;
          }
          //End t
        }
        //End coeff
      }
      //End tilt2
    }
    //End tilt1
  }
  //End iff
  return _SUCCESS_;
}


/**
 * Small helper function to get bessel integrals for every value of t
 * in a parrallelized code for integrated contributions
 *
 * @param pba                  Input: pointer to background struct
 * @param pma                  Input: pointer to matter struct
 * @param index_l              Input: l index
 * @param pmw                  Input: pointer to matter workspace
 * @return the error status
 */
int matter_get_bessel_fort_parallel_integrated(
                      struct matters* pma,
                      int index_l,
                      struct matters_workspace* pmw
                      ){
  double** window_bessel_real = pmw->window_bessel_real[index_l];
  double** window_bessel_imag = pmw->window_bessel_imag[index_l];
  int index_l_eval = (pma->uses_bessel_storeall?pma->l_sampling[index_l]:index_l);
  short integrate_logarithmically = (pma->uses_integration == matter_integrate_tw_logt);
  int index_coeff;
  double res_real,res_imag;
  int last_t = 0;
  double t_min = 0.0,t_max = 1.0;
  int index_tilt1,index_tilt2,index_tilt1_tilt2;
  double a,b,h;
  double* bi_real_i;
  double* bi_imag_i;
  double* ddbi_real_i;
  double* ddbi_imag_i;
  if(pma->has_integrated_windows){
    class_test(!pma->has_tilt_reduced,
               pma->error_message,
               "Has integrated windows, but not reduced tilt. This is a bug.");
    for(index_tilt1=0;index_tilt1<pma->tilt_size;++index_tilt1){
      for(index_tilt2=index_tilt1;index_tilt2<pma->tilt_size;++index_tilt2){
        index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);
        t_max = 1.0-pma->bi_maximal_t_offset;//186-pma->bi_maximal_t_offset found important;
        t_min = 0.0+pma->bi_maximal_t_offset;//186+pma->bi_maximal_t_offset;
        double y_min,y_max;
        y_min = -log(1-t_min);
        y_max = -log(1-t_max);//+pma->bi_maximal_t_offset);
        int index_t;
        double t,eval_pt=0.0;
        for(index_coeff=0;index_coeff<pma->size_fft_cutoff;++index_coeff){
          last_t = pma->bi_size[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff]-2;
          bi_real_i = pma->bi_real[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          bi_imag_i = pma->bi_imag[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          ddbi_real_i = pma->ddbi_real[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          ddbi_imag_i = pma->ddbi_imag[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff];
          // HERE :: last_t t change
          last_t = 0;
          for(index_t=0;index_t<pma->t_size;++index_t){
            matter_get_t_orig(index_t);
            eval_pt = 1.0-t;
            if(eval_pt>pma->bi_max[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff]){
              window_bessel_real[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t+pma->t_size] = 0.0;
              window_bessel_imag[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t+pma->t_size] = 0.0;
              continue;
            }
            class_call(matter_spline_hunt(pma->bi_sampling,
                                          pma->bi_size[index_tilt1_tilt2][index_l_eval*pma->size_fft_result+index_coeff],
                                          eval_pt,
                                          &last_t,
                                          &h,
                                          &a,
                                          &b,
                                          pma->error_message),
                         pma->error_message,
                         pma->error_message);
            res_real =a * bi_real_i[last_t] + b * bi_real_i[last_t+1] + ((a*a*a-a)* ddbi_real_i[last_t] + (b*b*b-b)* ddbi_real_i[last_t+1])*h*h/6.;
            res_imag =a * bi_imag_i[last_t] + b * bi_imag_i[last_t+1] + ((a*a*a-a)* ddbi_imag_i[last_t] + (b*b*b-b)* ddbi_imag_i[last_t+1])*h*h/6.;
            window_bessel_real[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t+pma->t_size] = res_real;
            window_bessel_imag[index_tilt1_tilt2*pma->size_fft_result+index_coeff][index_t+pma->t_size] = res_imag;
          }
          //End t
        }
        //End coeff
      }
      //End tilt2
    }
    //End tilt1
  }
  //End iff
  return _SUCCESS_;
}


/**
 * Small helper function for preparing an interpolation hunt,
 * searching with binary search for the starting position of the hunt.
 *
 * @param x_array               Input: pointer to x array
 * @param x_size                Input: size of x array
 * @param x                     Input: x position to search at
 * @param last                  Input/Output: last found position
 * @param err_msg               Input/Output: Error messages
 * @return the error status
 */
int matter_spline_prepare_hunt(
  double* x_array,
  int x_size,
  double x,
  int* last,
  ErrorMsg errmsg){
  int inf,sup,mid;

  inf=0;
  sup=x_size-1;

  if (x_array[inf] < x_array[sup]){

    if (x < x_array[inf]) {
      *last = inf;
      return _SUCCESS_;
    }

    if (x > x_array[sup]) {
      *last = sup;
      return _SUCCESS_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < x_array[sup]) {
      *last = sup;
      return _SUCCESS_;
    }

    if (x > x_array[inf]) {
      *last = inf;
      return _SUCCESS_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  *last = inf;
  return _SUCCESS_;
}


/**
 * Small helper function for doing the interpolation hunt without
 * returning the final array
 *
 * Starts searching at last found position and gradually increases
 * stepsize of search
 *
 * Returns the found interval of the x as the index last in the array,
 * the width of the interval h, and the parameters a,b quantifying the
 * relative distance x along the interval
 * 
 * a=1 if x is on the left border, and a=0 if on the right border
 * b=1-a
 * @param x_array               Input: pointer to x array
 * @param x_size                Input: size of x array
 * @param x                     Input: x position to search at
 * @param last                  Input/Output: last found position
 * @param h                     Output: Width of interval of found x
 * @param a                     Output: Relative distance along interval
 * @param b                     Output: 1-a
 * @param err_msg               Input/Output: Error messages
 * @return the error status
 */
int matter_spline_hunt(
  double* x_array,
  int x_size,
  double x,
  int* last,
  double* h,
  double* a,
  double* b,
  ErrorMsg errmsg
  ){
  int last_index = *last;
  //Old :: last_index = x_size-1 (and if >=x_size) , but here inf = last_index,sup=last_index+1 => x_size => eval of array AT x_size, which is disallowed
  if(last_index>=x_size-1){last_index=x_size-2;}
  if(last_index<0){last_index=0;}
  int inf,sup,mid,inc;
  inc=1;
  if (x >= x_array[last_index]) {
    if (x > x_array[x_size-1]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,
        x,x_array[x_size-1]);
      return _FAILURE_;
    }
    /* try closest neighboor upward */
    inf = last_index;
    sup = inf + inc;
    if (x > x_array[sup]) {
      /* hunt upward */
      while (x > x_array[sup]) {
        inf = sup;
        inc += 1;
        sup += inc;
        if (sup > x_size-1) {
          sup = x_size-1;
        }
      }
      /* bisect */
      while (sup-inf > 1) {
        mid=(int)(0.5*(inf+sup));
        if (x < x_array[mid]) {sup=mid;}
        else {inf=mid;}
      }
    }
  }
  else {
    if (x < x_array[0]) {
      sprintf(errmsg,"%s(L:%d) : x=%.20e < x_min=%.20e",__func__,__LINE__,
        x,x_array[0]);
      return _FAILURE_;
    }
    /* try closest neighboor downward */
    sup = last_index;
    inf = sup - inc;
    if (x < x_array[inf]) {
      /* hunt downward */
      while (x < x_array[inf]) {
        sup = inf;
        inc += 1;
        inf -= inc;
        if (inf < 0) {
          inf = 0;
        }
      }
      /* bisect */
      while (sup-inf > 1) {
        mid=(int)(0.5*(inf+sup));
        if (x < x_array[mid]) {sup=mid;}
        else {inf=mid;}
      }
    }
  }
  last_index = inf;
  *last = last_index;
  *h = x_array[sup] - x_array[inf];
  *b = (x-x_array[inf])/(*h);
  *a = 1.0-(*b);
  return _SUCCESS_;
}

/**
 * Small helper function for swapping around the indices in the matter_workspace
 *
 * @param pmw               Input: pointer to matter workspace structure
 * @return the error status
 */
int matter_swap_workspace(struct matters_workspace* pmw){
  int temp;
  temp = pmw->index_wd2;
  pmw->index_wd2 = pmw->index_wd1;
  pmw->index_wd1 = temp;
  temp = pmw->index_radtp1;
  pmw->index_radtp1 = pmw->index_radtp2;
  pmw->index_radtp2 = temp;
  temp = pmw->index_ic1;
  pmw->index_ic2 = pmw->index_ic1;
  pmw->index_ic1 = temp;
  temp = pmw->index_cltp1;
  pmw->index_cltp1 = pmw->index_cltp2;
  pmw->index_cltp2 = temp;
  return _SUCCESS_;
}


/**
 * Write the bessel integral file, including the header,
 * and the actual data
 *
 * @param pma               Input: pointer to matter structure
 * @return the error status
 */
int matter_write_bessel_integrals(struct matters* pma){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Writing bessel file \n");
  }
  /**
   * Define initial variables
   * */
  FILE* write_file;
  int index_tilt1_tilt2;
  int index_fft_l;


  /**
   * Open file to write
   * */
  write_file = fopen(pma->bessel_file_name,"wb");
  class_test(!write_file,
             pma->error_message,
             "Could not create file %s \n",pma->bessel_file_name);

  /**
   * Write header
   * */
  fwrite(&(pma->tilt_grid_size),sizeof(int),1,write_file);
  fwrite(&(pma->size_fft_result),sizeof(int),1,write_file);
  fwrite(&(pma->l_size_recursion),sizeof(int),1,write_file);
  fwrite(&(pma->bessel_recursion_t_size),sizeof(int),1,write_file);

  
  /**
   * Write sampling
   * */
  fwrite(pma->bi_sampling,sizeof(double),(pma->bessel_recursion_t_size+2),write_file);

  
  /**
   * Write actual data
   * */
  for(index_tilt1_tilt2=0;index_tilt1_tilt2<pma->tilt_grid_size;++index_tilt1_tilt2){
    for(index_fft_l=0;index_fft_l<pma->size_fft_result*pma->l_size_recursion;++index_fft_l){
      fwrite(&(pma->bi_size[index_tilt1_tilt2][index_fft_l]),sizeof(int),1,write_file);
      fwrite(&(pma->bi_max[index_tilt1_tilt2][index_fft_l]),sizeof(double),1,write_file);
      fwrite(pma->bi_real[index_tilt1_tilt2][index_fft_l],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_fft_l],write_file);
      fwrite(pma->bi_imag[index_tilt1_tilt2][index_fft_l],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_fft_l],write_file);
      fwrite(pma->ddbi_real[index_tilt1_tilt2][index_fft_l],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_fft_l],write_file);
      fwrite(pma->ddbi_imag[index_tilt1_tilt2][index_fft_l],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_fft_l],write_file);
    }
  }

  
  /**
   * Close the file
   * */
  fclose(write_file);
  return _SUCCESS_;
}


/**
 * Read the contents of the bessel file. Gives error if header
 * or contents are invalid.
 *
 * @param pma               Input: pointer to matter structure
 * @param is_correct_file  Output: pointer to correctness flag output
 * @return the error status
 */
int matter_read_bessel_integrals(struct matters* pma){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Reading bessel file content \n");
  }
  
  /**
   * Define initial variables
   * */
  int index_tilt1,index_tilt2,index_tilt1_tilt2;
  int index_l,index_coeff;
  int tilt_grid_size_temp;
  int fft_size_temp;
  int l_size_temp;
  int bessel_recursion_t_size_temp;
  int f_read;
#ifdef _OPENMP
  double read_start_omp = omp_get_wtime();
#else
  double read_start_omp = 0.0;
#endif
  FILE* read_file;

  
  /**
   * Open file for reading and check for errors during opening
   * */
  read_file = fopen(pma->bessel_file_name,"rb");
  class_test(!read_file,
             pma->error_message,
             "file '%s' missing/unopenable even though initial check indicated existence.",pma->bessel_file_name);


  /**
   * Read header
   * */
  f_read = 0;
  f_read+=fread(&tilt_grid_size_temp,sizeof(int),1,read_file);
  f_read+=fread(&fft_size_temp,sizeof(int),1,read_file);
  f_read+=fread(&l_size_temp,sizeof(int),1,read_file);
  f_read+=fread(&bessel_recursion_t_size_temp,sizeof(int),1,read_file);
  class_test(f_read!=4,
             pma->error_message,
             "file '%s' is corrupted even though initial check indicated none.",pma->bessel_file_name);


  /**
   * Allocate arrays to store content of file in
   * */
  class_alloc(pma->bi_real,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  class_alloc(pma->bi_imag,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  class_alloc(pma->bi_size,
              pma->tilt_grid_size*sizeof(int*),
              pma->error_message);
  class_alloc(pma->bi_max,
              pma->tilt_grid_size*sizeof(double*),
              pma->error_message);
  class_alloc(pma->bi_sampling,
              (pma->bessel_recursion_t_size+2)*sizeof(double),
              pma->error_message);
  class_alloc(pma->ddbi_real,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  class_alloc(pma->ddbi_imag,
              pma->tilt_grid_size*sizeof(double**),
              pma->error_message);
  for(index_tilt1=0;index_tilt1<pma->tilt_size;++index_tilt1){
    for(index_tilt2=index_tilt1;index_tilt2<pma->tilt_size;++index_tilt2){
      index_tilt1_tilt2 = index_symmetric_matrix(index_tilt1,index_tilt2,pma->tilt_size);
      class_alloc(pma->bi_real[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      class_alloc(pma->bi_imag[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      class_alloc(pma->bi_size[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(int),
                  pma->error_message);
      class_alloc(pma->bi_max[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double),
                  pma->error_message);
      class_alloc(pma->ddbi_real[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
      class_alloc(pma->ddbi_imag[index_tilt1_tilt2],
                  pma->l_size_recursion*pma->size_fft_result*sizeof(double*),
                  pma->error_message);
    }
  }

  
  /**
   * Check header correctness
   * */
  class_test(pma->tilt_grid_size!=tilt_grid_size_temp,
             pma->error_message,
             "Invalid file read (tilt_grid)");
  class_test(pma->size_fft_result!=fft_size_temp,
             pma->error_message,
             "Invalid file read (fft)");
  class_test(pma->l_size_recursion!=l_size_temp,
             pma->error_message,
             "Invalid file read (l_size)");
  class_test(pma->bessel_recursion_t_size!=bessel_recursion_t_size_temp,
             pma->error_message,
             "Invalid file read (t_size)");

  
  /**
   * Read t sampling
   * */
  f_read = 0;
  f_read+=fread(pma->bi_sampling,sizeof(double),(pma->bessel_recursion_t_size+2),read_file);
  class_test(f_read!=(pma->bessel_recursion_t_size+2),
             pma->error_message,
             "Invalid file read (bi_sampling)");

  
  /**
   * Read all content and for each iteration check if content is read correctly
   * */
  for(index_tilt1_tilt2=0;index_tilt1_tilt2<pma->tilt_grid_size;++index_tilt1_tilt2){
    for(index_l=0;index_l<pma->l_size_recursion;++index_l){
      for(index_coeff=0;index_coeff<pma->size_fft_result;++index_coeff){
        f_read=0;
        f_read+=fread(&(pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]),sizeof(int),1,read_file);
        f_read+=fread(&(pma->bi_max[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]),sizeof(double),1,read_file);
        class_alloc(pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                    (pma->bessel_recursion_t_size+2)*sizeof(double),
                    pma->error_message);
        class_alloc(pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                    (pma->bessel_recursion_t_size+2)*sizeof(double),
                    pma->error_message);
        class_alloc(pma->ddbi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                    pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                    pma->error_message);
        class_alloc(pma->ddbi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                    pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff]*sizeof(double),
                    pma->error_message);
        f_read+=fread(pma->bi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],read_file);
        f_read+=fread(pma->bi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],read_file);
        f_read+=fread(pma->ddbi_real[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],read_file);
        f_read+=fread(pma->ddbi_imag[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],sizeof(double),pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],read_file);
        class_test(f_read!=2+4*pma->bi_size[index_tilt1_tilt2][index_l*pma->size_fft_result+index_coeff],
                   pma->error_message,
                   "Invalid file read (bi_size,bi_max,bi_real,bi_imag,ddbi_real or ddbi_imag)");
      }
    }
  }

  
  /**
   * Close the file
   * */
  fclose(read_file);
#ifdef _OPENMP
  double read_end_omp = omp_get_wtime();
#else
  double read_end_omp = 0.0;
#endif
  if(pma->matter_verbose > MATTER_VERBOSITY_TIMING ){
    printf(" -> Reading bessel integrals took %f REAL seconds \n",read_end_omp-read_start_omp);
  }
  return _SUCCESS_;
}


/**
 * Read the usability of the bessel integral binary file.
 * Checks for existence of file. If file exists, and it
 * is readable as a binary file, and the header is correctly
 * readable, and the header agrees with the precision parameters
 * of the current run, the file is usable.
 *
 * Otherwise, it is not
 *
 * @param pma               Input: pointer to matter structure
 * @param is_correct_file  Output: pointer to correctness flag output
 * @return the error status
 */
int matter_read_bessel_file_correct(struct matters* pma,short* is_correct_file){
  if(pma->matter_verbose > MATTER_VERBOSITY_FUNCTIONS){
    printf("Method :: Reading bessel file existence/usability/correctness \n");
  }
  /**
   * Define initial variables
   * */
  FILE* read_file;
  int tilt_grid_size_temp;
  int fft_size_temp;
  int l_size_temp;
  int bessel_recursion_t_size_temp;
  int f_read;

  /**
   * Check if file is readable at all (exists)
   * */
  f_read = 0;
#ifdef BESSEL_DIR
  char temp[100];
  sprintf(pma->bessel_file_name,BESSEL_DIR);
  sprintf(temp,"/bessel_%i_%i_%i_%i.bin",pma->tilt_grid_size,pma->size_fft_result,pma->l_size_recursion,pma->bessel_recursion_t_size);
  strcat(pma->bessel_file_name,temp);
#else
  sprintf(pma->bessel_file_name,"output/bessel_%i_%i_%i_%i.bin",pma->tilt_grid_size,pma->size_fft_result,pma->l_size_recursion,pma->bessel_recursion_t_size);
#endif
  read_file = fopen(pma->bessel_file_name,"rb");
  if(!read_file){*is_correct_file=_FALSE_;return _SUCCESS_;}


  /**
   * Check if header is readable
   * */
  f_read+=fread(&tilt_grid_size_temp,sizeof(int),1,read_file);
  f_read+=fread(&fft_size_temp,sizeof(int),1,read_file);
  f_read+=fread(&l_size_temp,sizeof(int),1,read_file);
  f_read+=fread(&bessel_recursion_t_size_temp,sizeof(int),1,read_file);
  if(f_read!=4){*is_correct_file=_FALSE_;}

  
  /**
   * Check if header agrees with desired precision parameters
   * */
  if(pma->tilt_grid_size!=tilt_grid_size_temp){*is_correct_file=_FALSE_;}
  if(pma->size_fft_result!=fft_size_temp){*is_correct_file=_FALSE_;}
  if(pma->l_size_recursion!=l_size_temp){*is_correct_file=_FALSE_;}
  if(pma->bessel_recursion_t_size!=bessel_recursion_t_size_temp){*is_correct_file=_FALSE_;}


  /**
   * Close the file again
   * */
  fclose(read_file);
  return _SUCCESS_;
}


/**
 * Set the indices relevant to handling the different window functions,
 *  especially given the number of elements depending on cross-correlations
 *  between different cl types, like nCl's and sCl's
 *
 *
 * @param pma               Input: pointer to matter structure
 * @param ppt               Input: pointer to perturbs structure
 * @return the error status
 */
int matter_obtain_window_indices(struct matters* pma){
  int index_cltp1,index_cltp2,index_cltp1_cltp2;
  int index_wd1;
  pma->num_window_grid = (pma->num_windows*(pma->num_windows+1))/2;

  //x = "missing" elements on the off-diagonal
  int x = (pma->num_windows-1)-pma->non_diag;
  int x_grid = (x*(x+1))/2;

  class_alloc(pma->window_size,
              pma->cltp_grid_size*sizeof(int),
              pma->error_message);
  class_alloc(pma->window_index_start,
              pma->cltp_grid_size*sizeof(int*),
              pma->error_message);
  class_alloc(pma->window_index_end,
              pma->cltp_grid_size*sizeof(int*),
              pma->error_message);
  for(index_cltp1=0;index_cltp1<pma->cltp_size;++index_cltp1){
    for(index_cltp2=index_cltp1;index_cltp2<pma->cltp_size;++index_cltp2){
      index_cltp1_cltp2 = index_symmetric_matrix(index_cltp1,index_cltp2,pma->cltp_size);
      class_alloc(pma->window_index_start[index_cltp1_cltp2],
                  pma->num_windows*sizeof(int),
                  pma->error_message);
      class_alloc(pma->window_index_end[index_cltp1_cltp2],
                  pma->num_windows*sizeof(int),
                  pma->error_message);
      if(index_cltp1==index_cltp2){
      //[NS]: All below is modified for DESC
        pma->window_size[index_cltp1_cltp2]=(pma->num_windows_per_cltp[index_cltp1]*(pma->num_windows_per_cltp[index_cltp1]+1))/2; //pma->num_window_grid-x_grid;
        for(index_wd1=0;index_wd1<pma->num_windows;++index_wd1){
          pma->window_index_start[index_cltp1_cltp2][index_wd1]=index_wd1;
          pma->window_index_end[index_cltp1_cltp2][index_wd1]=pma->num_windows_per_cltp[index_cltp1]-1;//MIN(index_wd1+pma->non_diag,pma->num_windows-1);
        }
      }
      else{
        pma->window_size[index_cltp1_cltp2]=pma->num_windows_per_cltp[index_cltp1]*pma->num_windows_per_cltp[index_cltp2];//pma->num_windows*pma->num_windows-2*x_grid;
        for(index_wd1=0;index_wd1<pma->num_windows;++index_wd1){
          pma->window_index_start[index_cltp1_cltp2][index_wd1]=0;//MAX(0,index_wd1-pma->non_diag);
          pma->window_index_end[index_cltp1_cltp2][index_wd1]=pma->num_windows_per_cltp[index_cltp2]-1;//MIN(index_wd1+pma->non_diag,pma->num_windows-1);
        }
      }
    }
  }
  return _SUCCESS_;
}

int main(){
  struct matters ma;
  matter_init(&ma);
  matter_free(&ma);
}
