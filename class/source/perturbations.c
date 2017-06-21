/** @file perturbations.c Documented perturbation module
 *
 * Julien Lesgourgues, 23.09.2010
 *
 * Deals with the perturbation evolution.
 * This module has two purposes:
 *
 * - at the beginning; to initialize the perturbations, i.e. to
 * integrate the perturbation equations, and store temporarily the terms
 * contributing to the source functions as a function of conformal
 * time. Then, to perform a few manipulations of these terms in order to
 * infer the actual source functions \f$ S^{X} (k, \tau) \f$, and to
 * store them as a function of conformal time inside an interpolation
 * table.
 *
 * - at any time in the code; to evaluate the source functions at a
 * given conformal time (by interpolating within the interpolation
 * table).
 *
 * Hence the following functions can be called from other modules:
 *
 * -# perturb_init() at the beginning (but after background_init() and thermodynamics_init())
 * -# perturb_sources_at_tau() at any later time
 * -# perturb_free() at the end, when no more calls to perturb_sources_at_tau() are needed
 */

#include "perturbations.h"


/**
 * Source function \f$ S^{X} (k, \tau) \f$ at a given conformal time tau.
 *
 * Evaluate source functions at given conformal time tau by reading
 * the pre-computed table and interpolating.
 *
 * @param ppt        Input: pointer to perturbation structure containing interpolation tables
 * @param index_md   Input: index of requested mode
 * @param index_ic   Input: index of requested initial condition
 * @param index_type Input: index of requested source function type
 * @param tau        Input: any value of conformal time
 * @param psource    Output: vector (already allocated) of source function as a function of k
 * @return the error status
 */

int perturb_sources_at_tau(
                           struct perturbs * ppt,
                           int index_md,
                           int index_ic,
                           int index_type,
                           double tau,
                           double * psource
                           ) {


  /** Summary: */

  /** - interpolate in pre-computed table contained in ppt */
  class_call(array_interpolate_two_bis(ppt->tau_sampling,
                                       1,
                                       0,
                                       ppt->sources[index_md][index_ic*ppt->tp_size[index_md]+index_type],
                                       ppt->k_size[index_md],
                                       ppt->tau_size,
                                       tau,
                                       psource,
                                       ppt->k_size[index_md],
                                       ppt->error_message),
             ppt->error_message,
             ppt->error_message);


  return _SUCCESS_;
}

/**
 * Initialize the perturbs structure, and in particular the table of source functions.
 *
 * Main steps:
 *
 * - given the values of the flags describing which kind of
 *   perturbations should be considered (modes: scalar/vector/tensor,
 *   initial conditions, type of source functions needed...),
 *   initialize indices and wavenumber list
 *
 * - define the time sampling for the output source functions
 *
 * - for each mode (scalar/vector/tensor): initialize the indices of
 *   relevant perturbations, integrate the differential system,
 *   compute and store the source functions.
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param pth Input: pointer to thermodynamics structure
 * @param ppt Output: Initialized perturbation structure
 * @return the error status
 */

int perturb_init(
                 struct precision * ppr,
                 struct background * pba,
                 struct thermo * pth,
                 struct perturbs * ppt
                 ) {

  /** Summary: */

  /** - define local variables */

  /* running index for modes */
  int index_md;
  /* running index for initial conditions */
  int index_ic;
  /* running index for wavenumbers */
  int index_k;
  /* pointer to one struct perturb_workspace per thread (one if no openmp) */
  struct perturb_workspace ** pppw;
  /* number of threads (always one if no openmp) */
  int number_of_threads=1;
  /* index of the thread (always 0 if no openmp) */
  int thread=0;

  /* This code can be optionally compiled with the openmp option for parallel computation.
     Inside parallel regions, the use of the command "return" is forbidden.
     For error management, instead of "return _FAILURE_", we will set the variable below
     to "abort = _TRUE_". This will lead to a "return _FAILURE_" just after leaving the
     parallel region. */
  int abort;

  /* unsigned integer that will be set to the size of the workspace */
  size_t sz;

#ifdef _OPENMP
  /* instrumentation times */
  double tstart, tstop, tspent;
#endif

  /** - perform preliminary checks */

  if (ppt->has_perturbations == _FALSE_) {
    if (ppt->perturbations_verbose > 0)
      printf("No sources requested. Perturbation module skipped.\n");
    return _SUCCESS_;
  }
  else {
    if (ppt->perturbations_verbose > 0)
      printf("Computing sources\n");
  }

  class_test((ppt->gauge == synchronous) && (pba->has_cdm == _FALSE_),
             ppt->error_message,
             "In the synchronous gauge, it is not self-consistent to assume no CDM: the later is used to define the initial timelike hypersurface. You can either add a negligible amount of CDM or switch to newtonian gauge");

  class_test ((ppr->tight_coupling_approximation < first_order_MB) ||
              (ppr->tight_coupling_approximation > compromise_CLASS),
              ppt->error_message,
              "your tight_coupling_approximation is set to %d, out of range defined in perturbations.h",ppr->tight_coupling_approximation);

  class_test ((ppr->radiation_streaming_approximation < rsa_null) ||
              (ppr->radiation_streaming_approximation > rsa_none),
              ppt->error_message,
              "your radiation_streaming_approximation is set to %d, out of range defined in perturbations.h",ppr->radiation_streaming_approximation);

  if (pba->has_ur == _TRUE_) {

    class_test ((ppr->ur_fluid_approximation < ufa_mb) ||
                (ppr->ur_fluid_approximation > ufa_none),
                ppt->error_message,
                "your ur_fluid_approximation is set to %d, out of range defined in perturbations.h",ppr->ur_fluid_approximation);
  }

  if (pba->has_ncdm == _TRUE_) {

    class_test ((ppr->ncdm_fluid_approximation < ncdmfa_mb) ||
                (ppr->ncdm_fluid_approximation > ncdmfa_none),
                ppt->error_message,
                "your ncdm_fluid_approximation is set to %d, out of range defined in perturbations.h",ppr->ncdm_fluid_approximation);
  }

  if (pba->has_fld == _TRUE_) {

    class_test(pba->w0_fld+pba->wa_fld >= 0.,
               ppt->error_message,
               "So far, the fluid is meant to be negligible at early time, and not to be important for defining the initial conditions of other species. You are using parameters for which this assumption may break down, so maybe it's the case to fully implement the fluid in the initial condition routine");

    class_test((pba->w0_fld==-1.) && (pba->wa_fld==0.),
               ppt->error_message,
               "Your choice of a fluid with (w0,wa)=(-1,0) is not valid due to instabilities in the unphysical perturbations of such a fluid. Try instead with a plain cosmological constant");
// MI: changed < to <= to stop the case w0=-1 and wa <>0.0 and provide error message.  
    class_test(((pba->w0_fld + pba->wa_fld +1.0)*(pba->w0_fld+1.0)) <= 0.0,
               ppt->error_message,
               "w crosses -1 between the infinite past and today, and this would lead to divergent perturbation equations for the fluid.");

  }

  if (pba->has_dcdm == _TRUE_) {

    class_test((ppt->has_cdi == _TRUE_) || (ppt->has_bi == _TRUE_) || (ppt->has_nid == _TRUE_) || (ppt->has_niv == _TRUE_),
               ppt->error_message,
               "Non-adiabatic initial conditions not coded in presence of decaying dark matter");

  }

  class_test(ppt->has_vectors == _TRUE_,
             ppt->error_message,
             "Vectors not coded yet");

  if ((ppt->has_niv == _TRUE_) && (ppt->perturbations_verbose > 0)) {
    printf("Warning: the niv initial conditions in CLASS (and also in CAMB) should still be double-checked: if you want to do it and send feedback, you are welcome!\n");
  }

  if (ppt->has_tensors == _TRUE_) {

    ppt->evolve_tensor_ur = _FALSE_;
    ppt->evolve_tensor_ncdm = _FALSE_;

    switch (ppt->tensor_method) {

    case (tm_photons_only):
      break;

    case (tm_massless_approximation):
      if ((pba->has_ur == _TRUE_) || (pba->has_ncdm == _TRUE_))
        ppt->evolve_tensor_ur = _TRUE_;
      break;

    case (tm_exact):
      if (pba->has_ur == _TRUE_)
        ppt->evolve_tensor_ur = _TRUE_;
      if (pba->has_ncdm == _TRUE_)
        ppt->evolve_tensor_ncdm = _TRUE_;
      break;
    }
  }

  /** - initialize all indices and lists in perturbs structure using perturb_indices_of_perturbs() */

  class_call(perturb_indices_of_perturbs(ppr,
                                         pba,
                                         pth,
                                         ppt),
             ppt->error_message,
             ppt->error_message);


  if (ppt->z_max_pk > pth->z_rec) {

    class_test(ppt->has_cmb == _TRUE_,
               ppt->error_message,
               "You requested a very high z_pk=%e, higher than z_rec=%e. This works very well when you don't ask for a calculation of the CMB source function(s). Remove any CMB from your output and try e.g. with 'output=mTk' or 'output=mTk,vTk'",
               ppt->z_max_pk,
               pth->z_rec);

        class_test(ppt->has_source_delta_m == _TRUE_,
               ppt->error_message,
               "You requested a very high z_pk=%e, higher than z_rec=%e. This works very well when you ask only transfer functions, e.g. with 'output=mTk' or 'output=mTk,vTk'. But if you need the total matter (e.g. with 'mPk', 'dCl', etc.) there is an issue with the calculation of delta_m at very early times. By default, delta_m is a gauge-invariant variable (the density fluctuation in comoving gauge) and this quantity is hard to get accurately at very early times. The solution is to define delta_m as the density fluctuation in the current gauge, synchronous or newtonian. For the moment this must be done manually by commenting the line 'ppw->delta_m += 3. *ppw->pvecback[pba->index_bg_a]*ppw->pvecback[pba->index_bg_H] * ppw->theta_m/k2;' in perturb_sources(). In the future there will be an option for doing it in an easier way.",
               ppt->z_max_pk,
               pth->z_rec);

  }



  /** - define the common time sampling for all sources using
      perturb_timesampling_for_sources() */

  class_call(perturb_timesampling_for_sources(ppr,
                                              pba,
                                              pth,
                                              ppt),
             ppt->error_message,
             ppt->error_message);

  /** - if we want to store perturbations, write titles and allocate storage */
  class_call(perturb_prepare_output(pba,ppt),
             ppt->error_message,
             ppt->error_message);


  /** - create an array of workspaces in multi-thread case */

#ifdef _OPENMP

#pragma omp parallel
  {
    number_of_threads = omp_get_num_threads();
  }
#endif

  class_alloc(pppw,number_of_threads * sizeof(struct perturb_workspace *),ppt->error_message);

  /** - loop over modes (scalar, tensors, etc). For each mode: */

  for (index_md = 0; index_md < ppt->md_size; index_md++) {

    if (ppt->perturbations_verbose > 1)
      printf("Evolving mode %d/%d\n",index_md+1,ppt->md_size);

    abort = _FALSE_;

    sz = sizeof(struct perturb_workspace);

#pragma omp parallel                                             \
  shared(pppw,ppr,pba,pth,ppt,index_md,abort,number_of_threads)  \
  private(thread)                                                \
  num_threads(number_of_threads)

    {

#ifdef _OPENMP
      thread=omp_get_thread_num();
#endif

      /** - --> (a) create a workspace (one per thread in multi-thread case) */

      class_alloc_parallel(pppw[thread],sz,ppt->error_message);

      /** - --> (b) initialize indices of vectors of perturbations with perturb_indices_of_current_vectors() */

      class_call_parallel(perturb_workspace_init(ppr,
                                                 pba,
                                                 pth,
                                                 ppt,
                                                 index_md,
                                                 pppw[thread]),
                          ppt->error_message,
                          ppt->error_message);

    } /* end of parallel region */

    if (abort == _TRUE_) return _FAILURE_;

    /** - --> (c) loop over initial conditions and wavenumbers; for each of them, evolve perturbations and compute source functions with perturb_solve() */

    for (index_ic = 0; index_ic < ppt->ic_size[index_md]; index_ic++) {

      if (ppt->perturbations_verbose > 1)
        printf("Evolving ic %d/%d\n",index_ic+1,ppt->ic_size[index_md]);

        if (ppt->perturbations_verbose > 1)
          printf("evolving %d wavenumbers\n",ppt->k_size[index_md]);

      abort = _FALSE_;

#pragma omp parallel                                                    \
  shared(pppw,ppr,pba,pth,ppt,index_md,index_ic,abort,number_of_threads) \
  private(index_k,thread,tstart,tstop,tspent)                           \
  num_threads(number_of_threads)

      {

#ifdef _OPENMP
        thread=omp_get_thread_num();
        tspent=0.;
#endif

#pragma omp for schedule (dynamic)

        /* integrating backwards is slightly more optimal for parallel runs */
        //for (index_k = 0; index_k < ppt->k_size; index_k++) {
        for (index_k = ppt->k_size[index_md]-1; index_k >=0; index_k--) {

          if ((ppt->perturbations_verbose > 2) && (abort == _FALSE_)) {
            printf("evolving mode k=%e /Mpc  (%d/%d)",ppt->k[index_md][index_k],index_k+1,ppt->k_size[index_md]);
            if (pba->sgnK != 0)
              printf(" (for scalar modes, corresponds to nu=%e)",sqrt(ppt->k[index_md][index_k]*ppt->k[index_md][index_k]+pba->K)/sqrt(pba->sgnK*pba->K));
            printf("\n");
          }

#ifdef _OPENMP
          tstart = omp_get_wtime();
#endif

          class_call_parallel(perturb_solve(ppr,
                                            pba,
                                            pth,
                                            ppt,
                                            index_md,
                                            index_ic,
                                            index_k,
                                            pppw[thread]),
                              ppt->error_message,
                              ppt->error_message);

#ifdef _OPENMP
          tstop = omp_get_wtime();

          tspent += tstop-tstart;
#endif

#pragma omp flush(abort)

        } /* end of loop over wavenumbers */

#ifdef _OPENMP
        if (ppt->perturbations_verbose>1)
          printf("In %s: time spent in parallel region (loop over k's) = %e s for thread %d\n",
                 __func__,tspent,omp_get_thread_num());
#endif

      } /* end of parallel region */

      if (abort == _TRUE_) return _FAILURE_;

    } /* end of loop over initial conditions */

    abort = _FALSE_;

#pragma omp parallel                                    \
  shared(pppw,ppt,index_md,abort,number_of_threads)     \
  private(thread)                                       \
  num_threads(number_of_threads)

    {

#ifdef _OPENMP
      thread=omp_get_thread_num();
#endif

      class_call_parallel(perturb_workspace_free(ppt,index_md,pppw[thread]),
                          ppt->error_message,
                          ppt->error_message);

    } /* end of parallel region */

    if (abort == _TRUE_) return _FAILURE_;

  } /* end loop over modes */

  free(pppw);

  return _SUCCESS_;
}

/**
 * Free all memory space allocated by perturb_init().
 *
 * To be called at the end of each run, only when no further calls to
 * perturb_sources_at_tau() are needed.
 *
 * @param ppt Input: perturbation structure to be freed
 * @return the error status
 */

int perturb_free(
                 struct perturbs * ppt
                 ) {

  int index_md,index_ic,index_type;
  int filenum;

  if (ppt->has_perturbations == _TRUE_) {

    for (index_md = 0; index_md < ppt->md_size; index_md++) {

      for (index_ic = 0; index_ic < ppt->ic_size[index_md]; index_ic++) {

        for (index_type = 0; index_type < ppt->tp_size[index_md]; index_type++) {

          free(ppt->sources[index_md][index_ic*ppt->tp_size[index_md]+index_type]);

        }

      }

      free(ppt->sources[index_md]);

      free(ppt->k[index_md]);

    }

    free(ppt->tau_sampling);

    free(ppt->tp_size);

    free(ppt->ic_size);

    free(ppt->k);

    free(ppt->k_size_cmb);

    free(ppt->k_size_cl);

    free(ppt->k_size);

    free(ppt->sources);

    /** Stuff related to perturbations output: */

    /** - Free non-NULL pointers */
    if (ppt->index_k_output_values != NULL)
      free(ppt->index_k_output_values);

    for (filenum = 0; filenum<_MAX_NUMBER_OF_K_FILES_; filenum++){
      if (ppt->scalar_perturbations_data[filenum] != NULL)
        free(ppt->scalar_perturbations_data[filenum]);
      if (ppt->vector_perturbations_data[filenum] != NULL)
        free(ppt->vector_perturbations_data[filenum]);
      if (ppt->tensor_perturbations_data[filenum] != NULL)
        free(ppt->tensor_perturbations_data[filenum]);
    }

  }

  return _SUCCESS_;

}

/**
 * Initialize all indices and allocate most arrays in perturbs structure.
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param pth Input: pointer to thermodynamics structure
 * @param ppt Input/Output: Initialized perturbation structure
 * @return the error status
 */

int perturb_indices_of_perturbs(
                                struct precision * ppr,
                                struct background * pba,
                                struct thermo * pth,
                                struct perturbs * ppt
                                ) {

  /** Summary: */

  /** - define local variables */

  int index_type;
  int index_md;
  int index_ic;
  int index_type_common;

  /** - count modes (scalar, vector, tensor) and assign corresponding indices */

  index_md = 0;
  class_define_index(ppt->index_md_scalars,ppt->has_scalars,index_md,1);
  class_define_index(ppt->index_md_vectors,ppt->has_vectors,index_md,1);
  class_define_index(ppt->index_md_tensors,ppt->has_tensors,index_md,1);
  ppt->md_size = index_md;

  class_test(index_md == 0,
             ppt->error_message,
             "you should have at least one out of {scalars, vectors, tensors} !!!");

  /** - allocate array of number of types for each mode, ppt->tp_size[index_md] */

  class_alloc(ppt->tp_size,ppt->md_size*sizeof(int),ppt->error_message);

  /** - allocate array of number of initial conditions for each mode, ppt->ic_size[index_md] */

  class_alloc(ppt->ic_size,ppt->md_size*sizeof(int),ppt->error_message);

  /** - allocate array of arrays of source functions for each mode, ppt->source[index_md] */

  class_alloc(ppt->sources,ppt->md_size * sizeof(double *),ppt->error_message);

  /** - initialization of all flags to false (will eventually be set to true later) */

  ppt->has_cmb = _FALSE_;
  ppt->has_lss = _FALSE_;

  ppt->has_source_t = _FALSE_;
  ppt->has_source_p = _FALSE_;
  ppt->has_source_delta_m = _FALSE_;
  ppt->has_source_delta_g = _FALSE_;
  ppt->has_source_delta_b = _FALSE_;
  ppt->has_source_delta_cdm = _FALSE_;
  ppt->has_source_delta_dcdm = _FALSE_;
  ppt->has_source_delta_fld = _FALSE_;
  ppt->has_source_delta_scf = _FALSE_;
  ppt->has_source_delta_dr = _FALSE_;
  ppt->has_source_delta_ur = _FALSE_;
  ppt->has_source_delta_ncdm = _FALSE_;
  ppt->has_source_theta_m = _FALSE_;
  ppt->has_source_theta_g = _FALSE_;
  ppt->has_source_theta_b = _FALSE_;
  ppt->has_source_theta_cdm = _FALSE_;
  ppt->has_source_theta_dcdm = _FALSE_;
  ppt->has_source_theta_fld = _FALSE_;
  ppt->has_source_theta_scf = _FALSE_;
  ppt->has_source_theta_dr = _FALSE_;
  ppt->has_source_theta_ur = _FALSE_;
  ppt->has_source_theta_ncdm = _FALSE_;
  ppt->has_source_phi = _FALSE_;
  ppt->has_source_phi_prime = _FALSE_;
  ppt->has_source_phi_plus_psi = _FALSE_;
  ppt->has_source_psi = _FALSE_;

  /** - source flags and indices, for sources that all modes have in
      common (temperature, polarization, ...). For temperature, the
      term t2 is always non-zero, while other terms are non-zero only
      for scalars and vectors. For polarization, the term e is always
      non-zero, while the term b is only for vectors and tensors. */

  if (ppt->has_cl_cmb_temperature == _TRUE_) {
    ppt->has_source_t = _TRUE_;
    ppt->has_cmb = _TRUE_;
  }

  if (ppt->has_cl_cmb_polarization == _TRUE_) {
    ppt->has_source_p = _TRUE_;
    ppt->has_cmb = _TRUE_;
  }

  index_type = 0;
  class_define_index(ppt->index_tp_t2,ppt->has_source_t,index_type,1);
  class_define_index(ppt->index_tp_p,ppt->has_source_p,index_type,1);
  index_type_common = index_type;

  /* indices for perturbed recombination */

  class_define_index(ppt->index_tp_perturbed_recombination_delta_temp,ppt->has_perturbed_recombination,index_type,1);
  class_define_index(ppt->index_tp_perturbed_recombination_delta_chi,ppt->has_perturbed_recombination,index_type,1);




  /** - define k values with perturb_get_k_list() */

  class_call(perturb_get_k_list(ppr,
                                pba,
                                pth,
                                ppt),
             ppt->error_message,
             ppt->error_message);

  /** - loop over modes. Initialize flags and indices which are specific to each mode. */

  for (index_md = 0; index_md < ppt->md_size; index_md++) {

    /** - (a) scalars */

    if (_scalars_) {

      /** - --> source flags and indices, for sources that are specific to scalars */

      if ((ppt->has_cl_cmb_lensing_potential == _TRUE_) || (ppt->has_cl_lensing_potential)) {
        ppt->has_lss = _TRUE_;
        ppt->has_source_phi_plus_psi = _TRUE_;
      }

      if ((ppt->has_pk_matter == _TRUE_) || (ppt->has_nl_corrections_based_on_delta_m)) {
        ppt->has_lss = _TRUE_;
        ppt->has_source_delta_m = _TRUE_;
      }

      if (ppt->has_density_transfers == _TRUE_) {
        ppt->has_lss = _TRUE_;
        ppt->has_source_delta_g = _TRUE_;
        ppt->has_source_delta_b = _TRUE_;
        if (pba->has_cdm == _TRUE_)
          ppt->has_source_delta_cdm = _TRUE_;
        if (pba->has_dcdm == _TRUE_)
          ppt->has_source_delta_dcdm = _TRUE_;
        if (pba->has_fld == _TRUE_)
          ppt->has_source_delta_fld = _TRUE_;
        if (pba->has_scf == _TRUE_)
          ppt->has_source_delta_scf = _TRUE_;
        if (pba->has_ur == _TRUE_)
          ppt->has_source_delta_ur = _TRUE_;
        if (pba->has_dr == _TRUE_)
          ppt->has_source_delta_dr = _TRUE_;
        if (pba->has_ncdm == _TRUE_)
          ppt->has_source_delta_ncdm = _TRUE_;
        // Thanks to the following lines, (phi,psi) are also stored as sources
        // (Obtained directly in newtonian gauge, infereed from (h,eta) in synchronous gauge).
        // If density transfer functions are requested in the (default) CLASS format,
        // (phi, psi) will be appended to the delta_i's in the final output.
        ppt->has_source_phi = _TRUE_;
        ppt->has_source_psi = _TRUE_;
      }

      if (ppt->has_velocity_transfers == _TRUE_) {
        ppt->has_lss = _TRUE_;
        ppt->has_source_theta_g = _TRUE_;
        ppt->has_source_theta_b = _TRUE_;
        if ((pba->has_cdm == _TRUE_) && (ppt->gauge != synchronous))
          ppt->has_source_theta_cdm = _TRUE_;
        if (pba->has_dcdm == _TRUE_)
          ppt->has_source_theta_dcdm = _TRUE_;
        if (pba->has_fld == _TRUE_)
          ppt->has_source_theta_fld = _TRUE_;
        if (pba->has_scf == _TRUE_)
          ppt->has_source_theta_scf = _TRUE_;
        if (pba->has_ur == _TRUE_)
          ppt->has_source_theta_ur = _TRUE_;
        if (pba->has_dr == _TRUE_)
          ppt->has_source_theta_dr = _TRUE_;
        if (pba->has_ncdm == _TRUE_)
          ppt->has_source_theta_ncdm = _TRUE_;
      }

      if (ppt->has_cl_number_count == _TRUE_) {
        ppt->has_lss = _TRUE_;
        if (ppt->has_nc_density == _TRUE_) {
          ppt->has_source_delta_m = _TRUE_;
        }
        if (ppt->has_nc_rsd == _TRUE_) {
          ppt->has_source_theta_m = _TRUE_;
        }
        if (ppt->has_nc_lens == _TRUE_) {
          ppt->has_source_phi_plus_psi = _TRUE_;
        }
        if (ppt->has_nc_gr == _TRUE_) {
          ppt->has_source_phi = _TRUE_;
          ppt->has_source_psi = _TRUE_;
          ppt->has_source_phi_prime = _TRUE_;
          ppt->has_source_phi_plus_psi = _TRUE_;
        }
      }

      index_type = index_type_common;
      class_define_index(ppt->index_tp_t0,         ppt->has_source_t,         index_type,1);
      class_define_index(ppt->index_tp_t1,         ppt->has_source_t,         index_type,1);
      class_define_index(ppt->index_tp_delta_m,    ppt->has_source_delta_m,   index_type,1);
      class_define_index(ppt->index_tp_delta_g,    ppt->has_source_delta_g,   index_type,1);
      class_define_index(ppt->index_tp_delta_b,    ppt->has_source_delta_b,   index_type,1);
      class_define_index(ppt->index_tp_delta_cdm,  ppt->has_source_delta_cdm, index_type,1);
      class_define_index(ppt->index_tp_delta_dcdm, ppt->has_source_delta_dcdm,index_type,1);
      class_define_index(ppt->index_tp_delta_fld,  ppt->has_source_delta_fld, index_type,1);
      class_define_index(ppt->index_tp_delta_scf,  ppt->has_source_delta_scf, index_type,1);
      class_define_index(ppt->index_tp_delta_dr,   ppt->has_source_delta_dr, index_type,1);
      class_define_index(ppt->index_tp_delta_ur,   ppt->has_source_delta_ur,  index_type,1);
      class_define_index(ppt->index_tp_delta_ncdm1,ppt->has_source_delta_ncdm,index_type,pba->N_ncdm);
      class_define_index(ppt->index_tp_theta_m,    ppt->has_source_theta_m,   index_type,1);
      class_define_index(ppt->index_tp_theta_g,    ppt->has_source_theta_g,   index_type,1);
      class_define_index(ppt->index_tp_theta_b,    ppt->has_source_theta_b,   index_type,1);
      class_define_index(ppt->index_tp_theta_cdm,  ppt->has_source_theta_cdm, index_type,1);
      class_define_index(ppt->index_tp_theta_dcdm, ppt->has_source_theta_dcdm,index_type,1);
      class_define_index(ppt->index_tp_theta_fld,  ppt->has_source_theta_fld, index_type,1);
      class_define_index(ppt->index_tp_theta_scf,  ppt->has_source_theta_scf, index_type,1);
      class_define_index(ppt->index_tp_theta_dr,   ppt->has_source_theta_dr,  index_type,1);
      class_define_index(ppt->index_tp_theta_ur,   ppt->has_source_theta_ur,  index_type,1);
      class_define_index(ppt->index_tp_theta_ncdm1,ppt->has_source_theta_ncdm,index_type,pba->N_ncdm);
      class_define_index(ppt->index_tp_phi,        ppt->has_source_phi,       index_type,1);
      class_define_index(ppt->index_tp_phi_prime,  ppt->has_source_phi_prime, index_type,1);
      class_define_index(ppt->index_tp_phi_plus_psi,ppt->has_source_phi_plus_psi,index_type,1);
      class_define_index(ppt->index_tp_psi,        ppt->has_source_psi,       index_type,1);
      ppt->tp_size[index_md] = index_type;

      class_test(index_type == 0,
                 ppt->error_message,
                 "inconsistent input: you asked for scalars, so you should have at least one non-zero scalar source type (temperature, polarization, lensing/gravitational potential, ...). Please adjust your input.");

      /** - --> count scalar initial conditions (for scalars: ad, cdi, nid, niv; for tensors: only one) and assign corresponding indices */

      index_ic = 0;
      class_define_index(ppt->index_ic_ad, ppt->has_ad, index_ic,1);
      class_define_index(ppt->index_ic_bi, ppt->has_bi, index_ic,1);
      class_define_index(ppt->index_ic_cdi,ppt->has_cdi,index_ic,1);
      class_define_index(ppt->index_ic_nid,ppt->has_nid,index_ic,1);
      class_define_index(ppt->index_ic_niv,ppt->has_niv,index_ic,1);
      ppt->ic_size[index_md] = index_ic;

      class_test(index_ic == 0,
                 ppt->error_message,
                 "you should have at least one adiabatic or isocurvature initial condition...} !!!");

    }

    /** - (b) vectors */

    if (_vectors_) {

      /** - --> source flags and indices, for sources that are specific to vectors */

      index_type = index_type_common;
      class_define_index(ppt->index_tp_t1,ppt->has_source_t,index_type,1);
      ppt->tp_size[index_md] = index_type;

      /*
      class_test(index_type == 0,
                 ppt->error_message,
                 "inconsistent input: you asked for vectors, so you should have at least one non-zero vector source type (temperature or polarization). Please adjust your input.");
      */

      /** - --> initial conditions for vectors*/

      index_ic = 0;
      /* not coded yet */
      ppt->ic_size[index_md] = index_ic;

    }

    /** - (c) tensors */
    if (_tensors_) {

      /** - --> source flags and indices, for sources that are specific to tensors */

      index_type = index_type_common;
      /* nothing specific, unlike for vectors and scalars! */
      ppt->tp_size[index_md] = index_type;

      /*
      class_test(index_type == 0,
                 ppt->error_message,
                 "inconsistent input: you asked for tensors, so you should have at least one non-zero tensor source type (temperature or polarization). Please adjust your input.");
      */

      /** - --> only one initial condition for tensors*/

      index_ic = 0;
      class_define_index(ppt->index_ic_ten,_TRUE_,index_ic,1);
      ppt->ic_size[index_md] = index_ic;

    }

    /** - (d) for each mode, allocate array of arrays of source functions for each initial conditions and wavenumber, (ppt->source[index_md])[index_ic][index_type] */

    class_alloc(ppt->sources[index_md],
                ppt->ic_size[index_md] * ppt->tp_size[index_md] * sizeof(double *),
                ppt->error_message);

  }

  return _SUCCESS_;

}

/**
 * Define time sampling for source functions.
 *
 * For each type, compute the list of values of tau at which sources
 * will be sampled.  Knowing the number of tau values, allocate all
 * arrays of source functions.
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param pth Input: pointer to thermodynamics structure
 * @param ppt Input/Output: Initialized perturbation structure
 * @return the error status
 */

int perturb_timesampling_for_sources(
                                     struct precision * ppr,
                                     struct background * pba,
                                     struct thermo * pth,
                                     struct perturbs * ppt
                                     ) {

  /** Summary: */

  /** - define local variables */

  int counter;
  int index_md;
  int index_type;
  int index_ic;
  int last_index_back;
  int last_index_thermo;
  int first_index_back;
  int first_index_thermo;

  double tau;
  double tau_ini;
  double tau_lower;
  double tau_upper;
  double tau_mid;

  double timescale_source;
  double rate_thermo;
  double rate_isw_squared;
  double a_prime_over_a;
  double a_primeprime_over_a;
  double * pvecback;
  double * pvecthermo;

  /** - allocate background/thermodynamics vectors */

  class_alloc(pvecback,pba->bg_size_short*sizeof(double),ppt->error_message);
  class_alloc(pvecthermo,pth->th_size*sizeof(double),ppt->error_message);

  /** - first, just count the number of sampling points in order to allocate the array containing all values */

  /** - (a) if CMB requested, first sampling point = when the universe
      stops being opaque; otherwise, start sampling gravitational
      potential at recombination [however, if perturbed recombination
      is requested, we also need to start the system before
      recombination. Otherwise, the initial conditions for gas
      temperature and ionization fraction perturbations (delta_T = 1/3
      delta_b, delta_x_e) are not valid]. */

  if ((ppt->has_cmb == _TRUE_)||(ppt->has_perturbed_recombination == _TRUE_)) {

    /* using bisection, search time tau such that the ratio of thermo
       to Hubble time scales tau_c/tau_h=aH/kappa' is equal to
       start_sources_at_tau_c_over_tau_h */

    tau_lower = pth->tau_ini;

    class_call(background_at_tau(pba,
                                 tau_lower,
                                 pba->short_info,
                                 pba->inter_normal,
                                 &first_index_back,
                                 pvecback),
               pba->error_message,
               ppt->error_message);

    class_call(thermodynamics_at_z(pba,
                                   pth,
                                   1./pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                   pth->inter_normal,
                                   &first_index_thermo,
                                   pvecback,
                                   pvecthermo),
               pth->error_message,
               ppt->error_message);

    class_test(pvecback[pba->index_bg_a]*
               pvecback[pba->index_bg_H]/
               pvecthermo[pth->index_th_dkappa] >
               ppr->start_sources_at_tau_c_over_tau_h,
               ppt->error_message,
               "your choice of initial time for computing sources is inappropriate: it corresponds to an earlier time than the one at which the integration of thermodynamical variables started (tau=%g). You should increase either 'start_sources_at_tau_c_over_tau_h' or 'recfast_z_initial'\n",
               tau_lower);


    tau_upper = pth->tau_rec;

    class_call(background_at_tau(pba,
                                 tau_upper,
                                 pba->short_info,
                                 pba->inter_normal,
                                 &first_index_back,
                                 pvecback),
               pba->error_message,
               ppt->error_message);

    class_call(thermodynamics_at_z(pba,
                                   pth,
                                   1./pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                   pth->inter_normal,
                                   &first_index_thermo,
                                   pvecback,
                                   pvecthermo),
               pth->error_message,
               ppt->error_message);

    class_test(pvecback[pba->index_bg_a]*
               pvecback[pba->index_bg_H]/
               pvecthermo[pth->index_th_dkappa] <
               ppr->start_sources_at_tau_c_over_tau_h,
               ppt->error_message,
               "your choice of initial time for computing sources is inappropriate: it corresponds to a time after recombination. You should decrease 'start_sources_at_tau_c_over_tau_h'\n");

    tau_mid = 0.5*(tau_lower + tau_upper);

    while (tau_upper - tau_lower > ppr->tol_tau_approx) {

      class_call(background_at_tau(pba,
                                   tau_mid,
                                   pba->short_info,
                                   pba->inter_normal,
                                   &first_index_back,
                                   pvecback),
                 pba->error_message,
                 ppt->error_message);

      class_call(thermodynamics_at_z(pba,
                                     pth,
                                     1./pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                     pth->inter_normal,
                                     &first_index_thermo,
                                     pvecback,
                                     pvecthermo),
                 pth->error_message,
                 ppt->error_message);


      if (pvecback[pba->index_bg_a]*
          pvecback[pba->index_bg_H]/
          pvecthermo[pth->index_th_dkappa] >
          ppr->start_sources_at_tau_c_over_tau_h)

        tau_upper = tau_mid;
      else
        tau_lower = tau_mid;

      tau_mid = 0.5*(tau_lower + tau_upper);

    }

    tau_ini = tau_mid;

  }
  else {

    /* check the time corresponding to the highest redshift requested in output plus one */
    class_call(background_tau_of_z(pba,
                                   ppt->z_max_pk+1,
                                   &tau_ini),
               pba->error_message,
               ppt->error_message);

    /* obsolete: previous choice was to start always at recombination time */
    /* tau_ini = pth->tau_rec; */

    /* set values of first_index_back/thermo */
    class_call(background_at_tau(pba,
                                 tau_ini,
                                 pba->short_info,
                                 pba->inter_normal,
                                 &first_index_back,
                                 pvecback),
               pba->error_message,
               ppt->error_message);

    class_call(thermodynamics_at_z(pba,
                                   pth,
                                   1./pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                   pth->inter_normal,
                                   &first_index_thermo,
                                   pvecback,
                                   pvecthermo),
               pth->error_message,
               ppt->error_message);
  }

  /** - (b) next sampling point = previous + ppr->perturb_sampling_stepsize * timescale_source, where:
      - --> if CMB requested:
      timescale_source1 = \f$ |g/\dot{g}| = |\dot{\kappa}-\ddot{\kappa}/\dot{\kappa}|^{-1} \f$;
      timescale_source2 = \f$ |2\ddot{a}/a-(\dot{a}/a)^2|^{-1/2} \f$ (to sample correctly the late ISW effect; and
      timescale_source=1/(1/timescale_source1+1/timescale_source2); repeat till today.
      - --> if CMB not requested:
      timescale_source = 1/aH; repeat till today.
  */

  counter = 1;
  last_index_back = first_index_back;
  last_index_thermo = first_index_thermo;
  tau = tau_ini;

  while (tau < pba->conformal_age) {

    class_call(background_at_tau(pba,
                                 tau,
                                 pba->short_info,
                                 pba->inter_closeby,
                                 &last_index_back,
                                 pvecback),
               pba->error_message,
               ppt->error_message);

    class_call(thermodynamics_at_z(pba,
                                   pth,
                                   1./pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                   pth->inter_closeby,
                                   &last_index_thermo,
                                   pvecback,
                                   pvecthermo),
               pth->error_message,
               ppt->error_message);

    if (ppt->has_cmb == _TRUE_) {

      /* variation rate of thermodynamics variables */
      rate_thermo = pvecthermo[pth->index_th_rate];

      /* variation rate of metric due to late ISW effect (important at late times) */
      a_prime_over_a = pvecback[pba->index_bg_H] * pvecback[pba->index_bg_a];
      a_primeprime_over_a = pvecback[pba->index_bg_H_prime] * pvecback[pba->index_bg_a]
        + 2. * a_prime_over_a * a_prime_over_a;
      rate_isw_squared = fabs(2.*a_primeprime_over_a-a_prime_over_a*a_prime_over_a);

      /* compute rate */
      timescale_source = sqrt(rate_thermo*rate_thermo+rate_isw_squared);
    }
    else {
      /* variation rate given by Hubble time */
      a_prime_over_a = pvecback[pba->index_bg_H] * pvecback[pba->index_bg_a];

      timescale_source = a_prime_over_a;
    }

    /* check it is non-zero */
    class_test(timescale_source == 0.,
               ppt->error_message,
               "null evolution rate, integration is diverging");

    /* compute inverse rate */
    timescale_source = 1./timescale_source;

    class_test(fabs(ppr->perturb_sampling_stepsize*timescale_source/tau) < ppr->smallest_allowed_variation,
               ppt->error_message,
               "integration step =%e < machine precision : leads either to numerical error or infinite loop",ppr->perturb_sampling_stepsize*timescale_source);

    tau = tau + ppr->perturb_sampling_stepsize*timescale_source;
    counter++;

  }

  /** - --> infer total number of time steps, ppt->tau_size */
  ppt->tau_size = counter;

  /** - --> allocate array of time steps, ppt->tau_sampling[index_tau] */
  class_alloc(ppt->tau_sampling,ppt->tau_size * sizeof(double),ppt->error_message);

  /** - --> repeat the same steps, now filling the array with each tau value: */

  /** - --> (b.1.) first sampling point = when the universe stops being opaque */

  counter = 0;
  ppt->tau_sampling[counter]=tau_ini;

  /** - --> (b.2.) next sampling point = previous + ppr->perturb_sampling_stepsize * timescale_source, where
      timescale_source1 = \f$ |g/\dot{g}| = |\dot{\kappa}-\ddot{\kappa}/\dot{\kappa}|^{-1} \f$;
      timescale_source2 = \f$ |2\ddot{a}/a-(\dot{a}/a)^2|^{-1/2} \f$ (to sample correctly the late ISW effect; and
      timescale_source=1/(1/timescale_source1+1/timescale_source2); repeat till today.
      If CMB not requested:
      timescale_source = 1/aH; repeat till today.  */

  last_index_back = first_index_back;
  last_index_thermo = first_index_thermo;
  tau = tau_ini;

  while (tau < pba->conformal_age) {

    class_call(background_at_tau(pba,
                                 tau,
                                 pba->short_info,
                                 pba->inter_closeby,
                                 &last_index_back,
                                 pvecback),
               pba->error_message,
               ppt->error_message);

    class_call(thermodynamics_at_z(pba,
                                   pth,
                                   1./pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                   pth->inter_closeby,
                                   &last_index_thermo,
                                   pvecback,
                                   pvecthermo),
               pth->error_message,
               ppt->error_message);

    if (ppt->has_cmb == _TRUE_) {

      /* variation rate of thermodynamics variables */
      rate_thermo = pvecthermo[pth->index_th_rate];

      /* variation rate of metric due to late ISW effect (important at late times) */
      a_prime_over_a = pvecback[pba->index_bg_H] * pvecback[pba->index_bg_a];
      a_primeprime_over_a = pvecback[pba->index_bg_H_prime] * pvecback[pba->index_bg_a]
        + 2. * a_prime_over_a * a_prime_over_a;
      rate_isw_squared = fabs(2.*a_primeprime_over_a-a_prime_over_a*a_prime_over_a);

      /* compute rate */
      timescale_source = sqrt(rate_thermo*rate_thermo+rate_isw_squared);
    }
    else {
      a_prime_over_a = pvecback[pba->index_bg_H] * pvecback[pba->index_bg_a];
      timescale_source = a_prime_over_a;
    }

    /* check it is non-zero */
    class_test(timescale_source == 0.,
               ppt->error_message,
               "null evolution rate, integration is diverging");

    /* compute inverse rate */
    timescale_source = 1./timescale_source;

    class_test(fabs(ppr->perturb_sampling_stepsize*timescale_source/tau) < ppr->smallest_allowed_variation,
               ppt->error_message,
               "integration step =%e < machine precision : leads either to numerical error or infinite loop",ppr->perturb_sampling_stepsize*timescale_source);

    tau = tau + ppr->perturb_sampling_stepsize*timescale_source;
    counter++;
    ppt->tau_sampling[counter]=tau;

  }

  /** - last sampling point = exactly today */
  ppt->tau_sampling[counter] = pba->conformal_age;

  free(pvecback);
  free(pvecthermo);

  /** - loop over modes, initial conditions and types. For each of
      them, allocate array of source functions. */

  for (index_md = 0; index_md < ppt->md_size; index_md++) {
    for (index_ic = 0; index_ic < ppt->ic_size[index_md]; index_ic++) {
      for (index_type = 0; index_type < ppt->tp_size[index_md]; index_type++) {

        class_alloc(ppt->sources[index_md][index_ic*ppt->tp_size[index_md]+index_type],
                    ppt->k_size[index_md] * ppt->tau_size * sizeof(double),
                    ppt->error_message);

      }
    }
  }

  return _SUCCESS_;
}

/**
 * Define the number of comoving wavenumbers using the information
 * passed in the precision structure.
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to thermodynamics structure
 * @param ppt        Input: pointer to perturbation structure
 * @return the error status
 */

int perturb_get_k_list(
                        struct precision * ppr,
                        struct background * pba,
                        struct thermo * pth,
                        struct perturbs * ppt
                        ) {
  int index_k, index_k_output, index_mode;
  double k,k_min=0.,k_rec,step,tau1;
  double * k_max_cmb;
  double * k_max_cl;
  double k_max=0.;
  double scale2;
  double *tmp_k_list;
  int newk_size, index_newk, add_k_output_value;

  /** Summary: */

  class_test(ppr->k_step_transition == 0.,
             ppt->error_message,
             "stop to avoid division by zero");

  class_test(pth->rs_rec == 0.,
             ppt->error_message,
             "stop to avoid division by zero");

  /** - allocate arrays related to k list for each mode */

  class_alloc(ppt->k_size_cmb,
              ppt->md_size*sizeof(int),
              ppt->error_message);
  class_alloc(ppt->k_size_cl,
              ppt->md_size*sizeof(int),
              ppt->error_message);
  class_alloc(ppt->k_size,
              ppt->md_size*sizeof(int),
              ppt->error_message);
  class_alloc(ppt->k,
              ppt->md_size*sizeof(double*),
              ppt->error_message);

  class_calloc(k_max_cmb,
               ppt->md_size,
               sizeof(double),
               ppt->error_message);
  class_calloc(k_max_cl,
               ppt->md_size,
               sizeof(double),
               ppt->error_message);

  /** - scalar modes */

  if (ppt->has_scalars == _TRUE_) {

    /* first value */
    if (pba->sgnK == 0) {
      /* K<0 (flat)  : start close to zero */
      k_min=ppr->k_min_tau0/pba->conformal_age;
    }
    else if (pba->sgnK == -1) {
      /* K<0 (open)  : start close to sqrt(-K)
         (in transfer modules, for scalars, this will correspond to q close to zero;
         for vectors and tensors, this value is even smaller than the minimum necessary value) */
      k_min=sqrt(-pba->K+pow(ppr->k_min_tau0/pba->conformal_age/pth->angular_rescaling,2));

    }
    else if (pba->sgnK == 1) {
      /* K>0 (closed): start from q=sqrt(k2+(1+m)K) equal to 3sqrt(K), i.e. k=sqrt((8-m)K) */
      k_min = sqrt((8.-1.e-4)*pba->K);
    }

    /** - --> find k_max (as well as k_max_cmb[ppt->index_md_scalars], k_max_cl[ppt->index_md_scalars]) */

    k_rec = 2. * _PI_ / pth->rs_rec; /* comoving scale corresponding to sound horizon at recombination */

    k_max_cmb[ppt->index_md_scalars] = k_min;
    k_max_cl[ppt->index_md_scalars] = k_min;
    k_max = k_min;

    if (ppt->has_cls == _TRUE_) {

      /* find k_max_cmb[ppt->index_md_scalars] : */

      /* choose a k_max_cmb[ppt->index_md_scalars] corresponding to a wavelength on the last
         scattering surface seen today under an angle smaller than
         pi/lmax: this is equivalent to
         k_max_cl[ppt->index_md_scalars]*[comvoving.ang.diameter.distance] > l_max */

      k_max_cmb[ppt->index_md_scalars] = ppr->k_max_tau0_over_l_max*ppt->l_scalar_max
        /pba->conformal_age/pth->angular_rescaling;
      k_max_cl[ppt->index_md_scalars] = k_max_cmb[ppt->index_md_scalars];
      k_max     = k_max_cmb[ppt->index_md_scalars];

      /* find k_max_cl[ppt->index_md_scalars] : */

      /* if we need density/lensing Cl's, we must impose a stronger condition,
         such that the minimum wavelength on the shell corresponding
         to the center of smallest redshift bin is seen under an
         angle smaller than pi/lmax. So we must multiply our previous
         k_max_cl[ppt->index_md_scalars] by the ratio tau0/(tau0-tau[center of smallest
         redshift bin]). Note that we could do the same with the
         lensing potential if we needed a very precise C_l^phi-phi at
         large l. We don't do it by default, because the lensed ClT,
         ClE would be marginally affected. */

      if ((ppt->has_cl_number_count == _TRUE_) || (ppt->has_cl_lensing_potential == _TRUE_)) {

        class_call(background_tau_of_z(pba,
                                       ppt->selection_mean[0],
                                       &tau1),
                   pba->error_message,
                   ppt->error_message);

        k_max_cl[ppt->index_md_scalars] = MAX(k_max_cl[ppt->index_md_scalars],ppr->k_max_tau0_over_l_max*ppt->l_lss_max/(pba->conformal_age-tau1)); // to be very accurate we should use angular diameter distance to given redshift instead of comoving radius: would implement corrections depending on curvature
        k_max    = k_max_cl[ppt->index_md_scalars];
      }
    }

    /* find k_max: */

    if ((ppt->has_pk_matter == _TRUE_) || (ppt->has_density_transfers == _TRUE_) || (ppt->has_velocity_transfers == _TRUE_))
      k_max = MAX(k_max,ppt->k_max_for_pk);

    if (ppt->has_nl_corrections_based_on_delta_m == _TRUE_)
      k_max = MAX(k_max,ppr->halofit_min_k_max);

    /** - --> test that result for k_min, k_max make sense */

    class_test(k_min<0.,
               ppt->error_message,
               "buggy definition of k_min");

    class_test(k_max<0.,
               ppt->error_message,
               "buggy definition of k_max");

    class_test(k_max<k_min,
               ppt->error_message,
               "buggy definition of k_min and/or k_max");

    /* if K>0, the transfer function will be calculated for discrete
       integer values of nu=3,4,5,... where nu=sqrt(k2+(1+m)K) and
       m=0,1,2 for scalars/vectors/tensors. However we are free to
       define in the perturbation module some arbitrary values of k:
       later on, the transfer module will interpolate at values of k
       corresponding exactly to integer values of nu. Hence, apart
       from the value of k_min and the step size in the vicinity of
       k_min, we define exactly the same sampling in the three cases
       K=0, K<0, K>0 */

    /* allocate array with, for the moment, the largest possible size */
    class_alloc(ppt->k[ppt->index_md_scalars],
                ((int)((k_max_cmb[ppt->index_md_scalars]-k_min)/k_rec/MIN(ppr->k_step_super,ppr->k_step_sub))+
                 (int)(MAX(ppr->k_per_decade_for_pk,ppr->k_per_decade_for_bao)*log(k_max/k_min)/log(10.))+3)
                *sizeof(double),ppt->error_message);

    /* first value */

    index_k=0;
    k = k_min;
    ppt->k[ppt->index_md_scalars][index_k] = k;
    index_k++;

    /* values until k_max_cmb[ppt->index_md_scalars] */

    while (k < k_max_cmb[ppt->index_md_scalars]) {

      /* the linear step is not constant, it has a step-like shape,
         centered around the characteristic scale set by the sound
         horizon at recombination (associated to the comoving wavenumber
         k_rec) */

      step = (ppr->k_step_super
              + 0.5 * (tanh((k-k_rec)/k_rec/ppr->k_step_transition)+1.)
              * (ppr->k_step_sub-ppr->k_step_super)) * k_rec;

      /* there is one other thing to take into account in the step
         size. There are two other characteristic scales that matter for
         the sampling: the Hubble scale today, k0=a0H0, and eventually
         curvature scale sqrt(|K|). We define "scale2" as the sum of the
         squared Hubble radius and squared curvature radius. We need to
         increase the sampling for k<sqrt(scale2), in order to get the
         first mutipoles accurate enough. The formula below reduces it
         gradually in the k-->0 limit, by up to a factor 10. The actual
         stepsize is still fixed by k_step_super, this is just a
         reduction factor. */

      scale2 = pow(pba->a_today*pba->H0,2)+fabs(pba->K);

      step *= (k*k/scale2+1.)/(k*k/scale2+1./ppr->k_step_super_reduction);

      class_test(step / k < ppr->smallest_allowed_variation,
                 ppt->error_message,
                 "k step =%e < machine precision : leads either to numerical error or infinite loop",
                 step * k_rec);

      k += step;

      class_test(k <= ppt->k[ppt->index_md_scalars][index_k-1],
                 ppt->error_message,
                 "consecutive values of k should differ and should be in growing order");

      ppt->k[ppt->index_md_scalars][index_k] = k;

      index_k++;
    }

    ppt->k_size_cmb[ppt->index_md_scalars] = index_k;

    /* values until k_max_cl[ppt->index_md_scalars] */

    while (k < k_max_cl[ppt->index_md_scalars]) {

      k *= pow(10.,1./(ppr->k_per_decade_for_pk
                       +(ppr->k_per_decade_for_bao-ppr->k_per_decade_for_pk)
                       *(1.-tanh(pow((log(k)-log(ppr->k_bao_center*k_rec))/log(ppr->k_bao_width),4)))));

      ppt->k[ppt->index_md_scalars][index_k] = k;
      index_k++;
    }

    ppt->k_size_cl[ppt->index_md_scalars] = index_k;

    /* values until k_max */

    while (k < k_max) {

      k *= pow(10.,1./(ppr->k_per_decade_for_pk
                       +(ppr->k_per_decade_for_bao-ppr->k_per_decade_for_pk)
                       *(1.-tanh(pow((log(k)-log(ppr->k_bao_center*k_rec))/log(ppr->k_bao_width),4)))));

      ppt->k[ppt->index_md_scalars][index_k] = k;
      index_k++;
    }

    ppt->k_size[ppt->index_md_scalars] = index_k;

    class_realloc(ppt->k[ppt->index_md_scalars],
                  ppt->k[ppt->index_md_scalars],
                  ppt->k_size[ppt->index_md_scalars]*sizeof(double),
                  ppt->error_message);
  }

  /** - vector modes */

  if (ppt->has_vectors == _TRUE_) {

    /* first value */
    if (pba->sgnK == 0) {
      /* K<0 (flat)  : start close to zero */
      k_min=ppr->k_min_tau0/pba->conformal_age;
    }
    else if (pba->sgnK == -1) {
      /* K<0 (open)  : start close to sqrt(-K)
         (in transfer modules, for scalars, this will correspond to q close to zero;
         for vectors and tensors, this value is even smaller than the minimum necessary value) */
      k_min=sqrt(-pba->K+pow(ppr->k_min_tau0/pba->conformal_age/pth->angular_rescaling,2));

    }
    else if (pba->sgnK == 1) {
      /* K>0 (closed): start from q=sqrt(k2+(1+m)K) equal to 3sqrt(K), i.e. k=sqrt((8-m)K) */
      k_min = sqrt((7.-1.e-4)*pba->K);
    }

    /** - --> find k_max (as well as k_max_cmb[ppt->index_md_vectors], k_max_cl[ppt->index_md_vectors]) */

    k_rec = 2. * _PI_ / pth->rs_rec; /* comoving scale corresponding to sound horizon at recombination */

    k_max_cmb[ppt->index_md_vectors] = k_min;
    k_max_cl[ppt->index_md_vectors] = k_min;
    k_max = k_min;

    if (ppt->has_cls == _TRUE_) {

      /* find k_max_cmb: */

      /* choose a k_max_cmb corresponding to a wavelength on the last
         scattering surface seen today under an angle smaller than
         pi/lmax: this is equivalent to
         k_max_cl*[comvoving.ang.diameter.distance] > l_max */

      k_max_cmb[ppt->index_md_vectors] = ppr->k_max_tau0_over_l_max*ppt->l_vector_max
        /pba->conformal_age/pth->angular_rescaling;
      k_max_cl[ppt->index_md_vectors]  = k_max_cmb[ppt->index_md_vectors];
      k_max     = k_max_cmb[ppt->index_md_vectors];
    }

    /** - --> test that result for k_min, k_max make sense */

    class_test(k_min<0.,
               ppt->error_message,
               "buggy definition of k_min");

    class_test(k_max<0.,
               ppt->error_message,
               "buggy definition of k_max");

    class_test(k_max<k_min,
               ppt->error_message,
               "buggy definition of k_min and/or k_max");

    /* if K>0, the transfer function will be calculated for discrete
       integer values of nu=3,4,5,... where nu=sqrt(k2+(1+m)K) and
       m=0,1,2 for scalars/vectors/tensors. However we are free to
       define in the perturbation module some arbitrary values of k:
       later on, the transfer module will interpolate at values of k
       corresponding exactly to integer values of nu. Hence, apart
       from the value of k_min and the step size in the vicinity of
       k_min, we define exactly the same sampling in the three cases
       K=0, K<0, K>0 */

    /* allocate array with, for the moment, the largest possible size */
    class_alloc(ppt->k[ppt->index_md_vectors],
                ((int)((k_max_cmb[ppt->index_md_vectors]-k_min)/k_rec/MIN(ppr->k_step_super,ppr->k_step_sub))+1)
                *sizeof(double),ppt->error_message);

    /* first value */

    index_k=0;
    k = k_min;
    ppt->k[ppt->index_md_vectors][index_k] = k;
    index_k++;

    /* values until k_max_cmb[ppt->index_md_vectors] */

    while (k < k_max_cmb[ppt->index_md_vectors]) {

      /* the linear step is not constant, it has a step-like shape,
         centered around the characteristic scale set by the sound
         horizon at recombination (associated to the comoving wavenumber
         k_rec) */

      step = (ppr->k_step_super
              + 0.5 * (tanh((k-k_rec)/k_rec/ppr->k_step_transition)+1.)
              * (ppr->k_step_sub-ppr->k_step_super)) * k_rec;

      /* there is one other thing to take into account in the step
         size. There are two other characteristic scales that matter for
         the sampling: the Hubble scale today, k0=a0H0, and eventually
         curvature scale sqrt(|K|). We define "scale2" as the sum of the
         squared Hubble radius and squared curvature radius. We need to
         increase the sampling for k<sqrt(scale2), in order to get the
         first mutipoles accurate enough. The formula below reduces it
         gradually in the k-->0 limit, by up to a factor 10. The actual
         stepsize is still fixed by k_step_super, this is just a
         reduction factor. */

      scale2 = pow(pba->a_today*pba->H0,2)+fabs(pba->K);

      step *= (k*k/scale2+1.)/(k*k/scale2+1./ppr->k_step_super_reduction);

      class_test(step / k < ppr->smallest_allowed_variation,
                 ppt->error_message,
                 "k step =%e < machine precision : leads either to numerical error or infinite loop",
                 step * k_rec);

      k += step;

      class_test(k <= ppt->k[ppt->index_md_scalars][index_k-1],
                 ppt->error_message,
                 "consecutive values of k should differ and should be in growing order");

      ppt->k[ppt->index_md_vectors][index_k] = k;

      index_k++;
    }

    ppt->k_size_cmb[ppt->index_md_vectors] = index_k;
    ppt->k_size_cl[ppt->index_md_vectors] = index_k;
    ppt->k_size[ppt->index_md_vectors] = index_k;

    class_realloc(ppt->k[ppt->index_md_vectors],
                  ppt->k[ppt->index_md_vectors],
                  ppt->k_size[ppt->index_md_vectors]*sizeof(double),
                  ppt->error_message);
  }

  /** - tensor modes */

  if (ppt->has_tensors == _TRUE_) {

    /* first value */
    if (pba->sgnK == 0) {
      /* K<0 (flat)  : start close to zero */
      k_min=ppr->k_min_tau0/pba->conformal_age;
    }
    else if (pba->sgnK == -1) {
      /* K<0 (open)  : start close to sqrt(-K)
         (in transfer modules, for scalars, this will correspond to q close to zero;
         for vectors and tensors, this value is even smaller than the minimum necessary value) */
      k_min=sqrt(-pba->K+pow(ppr->k_min_tau0/pba->conformal_age/pth->angular_rescaling,2));

    }
    else if (pba->sgnK == 1) {
      /* K>0 (closed): start from q=sqrt(k2+(1+m)K) equal to 3sqrt(K), i.e. k=sqrt((8-m)K) */
      k_min = sqrt((6.-1.e-4)*pba->K);
    }

    /** - --> find k_max (as well as k_max_cmb[ppt->index_md_tensors], k_max_cl[ppt->index_md_tensors]) */

    k_rec = 2. * _PI_ / pth->rs_rec; /* comoving scale corresponding to sound horizon at recombination */

    k_max_cmb[ppt->index_md_tensors] = k_min;
    k_max_cl[ppt->index_md_tensors] = k_min;
    k_max = k_min;

    if (ppt->has_cls == _TRUE_) {

      /* find k_max_cmb[ppt->index_md_tensors]: */

      /* choose a k_max_cmb[ppt->index_md_tensors] corresponding to a wavelength on the last
         scattering surface seen today under an angle smaller than
         pi/lmax: this is equivalent to
         k_max_cl[ppt->index_md_tensors]*[comvoving.ang.diameter.distance] > l_max */

      k_max_cmb[ppt->index_md_tensors] = ppr->k_max_tau0_over_l_max*ppt->l_tensor_max
        /pba->conformal_age/pth->angular_rescaling;
      k_max_cl[ppt->index_md_tensors]  = k_max_cmb[ppt->index_md_tensors];
      k_max     = k_max_cmb[ppt->index_md_tensors];
    }

    /** - --> test that result for k_min, k_max make sense */

    class_test(k_min<0.,
               ppt->error_message,
               "buggy definition of k_min");

    class_test(k_max<0.,
               ppt->error_message,
               "buggy definition of k_max");

    class_test(k_max<k_min,
               ppt->error_message,
               "buggy definition of k_min and/or k_max");

    /* if K>0, the transfer function will be calculated for discrete
       integer values of nu=3,4,5,... where nu=sqrt(k2+(1+m)K) and
       m=0,1,2 for scalars/vectors/tensors. However we are free to
       define in the perturbation module some arbitrary values of k:
       later on, the transfer module will interpolate at values of k
       corresponding exactly to integer values of nu. Hence, apart
       from the value of k_min and the step size in the vicinity of
       k_min, we define exactly the same sampling in the three cases
       K=0, K<0, K>0 */

    /* allocate array with, for the moment, the largest possible size */
    class_alloc(ppt->k[ppt->index_md_tensors],
                ((int)((k_max_cmb[ppt->index_md_tensors]-k_min)/k_rec/MIN(ppr->k_step_super,ppr->k_step_sub))+1)
                *sizeof(double),ppt->error_message);

    /* first value */

    index_k=0;
    k = k_min;
    ppt->k[ppt->index_md_tensors][index_k] = k;
    index_k++;

    /* values until k_max_cmb[ppt->index_md_tensors] */

    while (k < k_max_cmb[ppt->index_md_tensors]) {

      /* the linear step is not constant, it has a step-like shape,
         centered around the characteristic scale set by the sound
         horizon at recombination (associated to the comoving wavenumber
         k_rec) */

      step = (ppr->k_step_super
              + 0.5 * (tanh((k-k_rec)/k_rec/ppr->k_step_transition)+1.)
              * (ppr->k_step_sub-ppr->k_step_super)) * k_rec;

      /* there is one other thing to take into account in the step
         size. There are two other characteristic scales that matter for
         the sampling: the Hubble scale today, k0=a0H0, and eventually
         curvature scale sqrt(|K|). We define "scale2" as the sum of the
         squared Hubble radius and squared curvature radius. We need to
         increase the sampling for k<sqrt(scale2), in order to get the
         first mutipoles accurate enough. The formula below reduces it
         gradually in the k-->0 limit, by up to a factor 10. The actual
         stepsize is still fixed by k_step_super, this is just a
         reduction factor. */

      scale2 = pow(pba->a_today*pba->H0,2)+fabs(pba->K);

      step *= (k*k/scale2+1.)/(k*k/scale2+1./ppr->k_step_super_reduction);

      class_test(step / k < ppr->smallest_allowed_variation,
                 ppt->error_message,
                 "k step =%e < machine precision : leads either to numerical error or infinite loop",
                 step * k_rec);

      k += step;

      class_test(k <= ppt->k[ppt->index_md_tensors][index_k-1],
                 ppt->error_message,
                 "consecutive values of k should differ and should be in growing order");

      ppt->k[ppt->index_md_tensors][index_k] = k;

      index_k++;
    }

    ppt->k_size_cmb[ppt->index_md_tensors] = index_k;
    ppt->k_size_cl[ppt->index_md_tensors] = index_k;
    ppt->k_size[ppt->index_md_tensors] = index_k;

    class_realloc(ppt->k[ppt->index_md_tensors],
                  ppt->k[ppt->index_md_tensors],
                  ppt->k_size[ppt->index_md_tensors]*sizeof(double),
                  ppt->error_message);
  }

  /** - If user asked for k_output_values, add those to all k lists: */
  if (ppt->k_output_values_num>0){
    /* Allocate storage */
    class_alloc(ppt->index_k_output_values,sizeof(double)*ppt->md_size*ppt->k_output_values_num,ppt->error_message);

    /** - --> Find indices in ppt->k[index_md] corresponding to 'k_output_values'.
        We are assuming that ppt->k is sorted and growing, and we have made sure
        that ppt->k_output_values is also sorted and growing.*/
    for (index_mode=0; index_mode<ppt->md_size; index_mode++){

      newk_size = ppt->k_size[index_mode]+ppt->k_output_values_num;

      class_alloc(tmp_k_list,sizeof(double)*newk_size,ppt->error_message);

      index_k=0;
      index_k_output=0;
      for (index_newk=0; index_newk<newk_size; index_newk++){
        /** - --> Decide if we should add k_output_value now. This has to be this complicated, since we
            can only compare the k-values when both indices are in range.*/
        if (index_k >= ppt->k_size[index_mode])
          add_k_output_value = _TRUE_;
        else if (index_k_output >= ppt->k_output_values_num)
          add_k_output_value = _FALSE_;
        else if (ppt->k_output_values[index_k_output] < ppt->k[index_mode][index_k])
          add_k_output_value = _TRUE_;
        else
          add_k_output_value = _FALSE_;

        if (add_k_output_value == _TRUE_){
          tmp_k_list[index_newk] = ppt->k_output_values[index_k_output];
          ppt->index_k_output_values[index_mode*ppt->k_output_values_num+index_k_output]=index_newk;
          index_k_output++;
        }
        else{
          tmp_k_list[index_newk] = ppt->k[index_mode][index_k];
          index_k++;
        }
      }

      free(ppt->k[index_mode]);
      ppt->k[index_mode] = tmp_k_list;
      ppt->k_size[index_mode] = newk_size;

      index_k = newk_size-1;
      while (ppt->k[index_mode][index_k] > k_max_cl[index_mode])
        index_k--;
      ppt->k_size_cl[index_mode] = MIN(index_k+2,ppt->k_size[index_mode]);

      index_k = newk_size-1;
      while (ppt->k[index_mode][index_k] > k_max_cmb[index_mode])
        index_k--;
      ppt->k_size_cmb[index_mode] = MIN(index_k+2,ppt->k_size[index_mode]);

      /** - --> The two MIN statements are here because in a normal run, the cl and cmb
          arrays contain a single k value larger than their respective k_max.
          We are mimicking this behavior. */
    }
  }

  /* For testing, can be useful to print the k list in a file:

  FILE * out=fopen("output/k","w");

  for (index_k=0; index_k < ppt->k_size[0]; index_k++) {

    fprintf(out,"%e\n",ppt->k[0][index_k],pba->K);

  }
     fclose(out);
  */

  /** - finally, find the global k_min and k_max for the ensemble of all modes 9scalars, vectors, tensors) */

  ppt->k_min = _HUGE_;
  ppt->k_max = 0.;
  if (ppt->has_scalars == _TRUE_) {
    ppt->k_min = MIN(ppt->k_min,ppt->k[ppt->index_md_scalars][0]); /* first value, inferred from perturbations structure */
    ppt->k_max = MAX(ppt->k_max,ppt->k[ppt->index_md_scalars][ppt->k_size[ppt->index_md_scalars]-1]); /* last value, inferred from perturbations structure */
  }
  if (ppt->has_vectors == _TRUE_) {
    ppt->k_min = MIN(ppt->k_min,ppt->k[ppt->index_md_vectors][0]); /* first value, inferred from perturbations structure */
    ppt->k_max = MAX(ppt->k_max,ppt->k[ppt->index_md_vectors][ppt->k_size[ppt->index_md_vectors]-1]); /* last value, inferred from perturbations structure */
  }
  if (ppt->has_tensors == _TRUE_) {
    ppt->k_min = MIN(ppt->k_min,ppt->k[ppt->index_md_tensors][0]); /* first value, inferred from perturbations structure */
    ppt->k_max = MAX(ppt->k_max,ppt->k[ppt->index_md_tensors][ppt->k_size[ppt->index_md_tensors]-1]); /* last value, inferred from perturbations structure */
  }

  free(k_max_cmb);
  free(k_max_cl);

  return _SUCCESS_;

}

/**
 * Initialize a perturb_workspace structure. All fields are allocated
 * here, with the exception of the perturb_vector '-->pv' field, which
 * is allocated separately in perturb_vector_init. We allocate one
 * such perturb_workspace structure per thread and per mode
 * (scalar/../tensor). Then, for each thread, all initial conditions
 * and wavenumbers will use the same workspace.
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param ppt        Input: pointer to the perturbation structure
 * @param index_md Input: index of mode under consideration (scalar/.../tensor)
 * @param ppw        Input/Output: pointer to perturb_workspace structure which fields are allocated or filled here
 * @return the error status
 */

int perturb_workspace_init(
                           struct precision * ppr,
                           struct background * pba,
                           struct thermo * pth,
                           struct perturbs * ppt,
                           int index_md,
                           struct perturb_workspace * ppw
                           ) {

  /** Summary: */

  /** - define local variables */

  int index_mt=0;
  int index_ap;
  int l;

  /** - Compute maximum l_max for any multipole */;
  if (_scalars_) {
    ppw->max_l_max = MAX(ppr->l_max_g, ppr->l_max_pol_g);
    if (pba->has_ur == _TRUE_) ppw->max_l_max = MAX(ppw->max_l_max, ppr->l_max_ur);
    if (pba->has_ncdm == _TRUE_) ppw->max_l_max = MAX(ppw->max_l_max, ppr->l_max_ncdm);
    if (pba->has_dr == _TRUE_) ppw->max_l_max = MAX(ppw->max_l_max, ppr->l_max_dr);
  }
  if (_tensors_) {
    ppw->max_l_max = MAX(ppr->l_max_g_ten, ppr->l_max_pol_g_ten);
    if (pba->has_ur == _TRUE_) ppw->max_l_max = MAX(ppw->max_l_max, ppr->l_max_ur);
    if (pba->has_ncdm == _TRUE_) ppw->max_l_max = MAX(ppw->max_l_max, ppr->l_max_ncdm);
  }

  /** - Allocate \f$ s_l\f$[ ] array for freestreaming of multipoles (see arXiv:1305.3261) and initialize
      to 1.0, which is the K=0 value. */
  class_alloc(ppw->s_l, sizeof(double)*(ppw->max_l_max+1),ppt->error_message);
  for (l=0; l<=ppw->max_l_max; l++){
    ppw->s_l[l] = 1.0;
  }

  /** - define indices of metric perturbations obeying constraint
      equations (this can be done once and for all, because the
      vector of metric perturbations is the same whatever the
      approximation scheme, unlike the vector of quantities to
      be integrated, which is allocated separately in
      perturb_vector_init) */

  if (_scalars_) {

    /* newtonian gauge */

    if (ppt->gauge == newtonian) {
      class_define_index(ppw->index_mt_psi,_TRUE_,index_mt,1); /* psi */
      class_define_index(ppw->index_mt_phi_prime,_TRUE_,index_mt,1); /* phi' */
    }

    /* synchronous gauge (note that eta is counted in the vector of
       quantities to be integrated, while here we only consider
       quantities obeying to constraint equations) */

    if (ppt->gauge == synchronous) {
      class_define_index(ppw->index_mt_h_prime,_TRUE_,index_mt,1);       /* h' */
      class_define_index(ppw->index_mt_h_prime_prime,_TRUE_,index_mt,1); /* h'' */
      class_define_index(ppw->index_mt_eta_prime,_TRUE_,index_mt,1);     /* eta' */
      class_define_index(ppw->index_mt_alpha,_TRUE_,index_mt,1);         /* alpha = (h' + 6 tau') / (2 k**2) */
      class_define_index(ppw->index_mt_alpha_prime,_TRUE_,index_mt,1);   /* alpha' */

    }

  }

  if (_vectors_) {

    /* newtonian gauge */

    if (ppt->gauge == newtonian) {

      class_define_index(ppw->index_mt_V_prime,_TRUE_,index_mt,1);

    }

    if (ppt->gauge == synchronous) {

      class_define_index(ppw->index_mt_hv_prime_prime,_TRUE_,index_mt,1);

    }

  }

  if (_tensors_) {
    class_define_index(ppw->index_mt_gw_prime_prime,_TRUE_,index_mt,1);
  }

  ppw->mt_size = index_mt;

  /** - allocate some workspace in which we will store temporarily the
      values of background, thermodynamics, metric and source
      quantities at a given time */

  class_alloc(ppw->pvecback,pba->bg_size_normal*sizeof(double),ppt->error_message);
  class_alloc(ppw->pvecthermo,pth->th_size*sizeof(double),ppt->error_message);
  class_alloc(ppw->pvecmetric,ppw->mt_size*sizeof(double),ppt->error_message);

  /** - count number of approximations, initialize their indices, and allocate their flags */
  index_ap=0;

  class_define_index(ppw->index_ap_tca,_TRUE_,index_ap,1);
  class_define_index(ppw->index_ap_rsa,_TRUE_,index_ap,1);

  if (_scalars_) {

    class_define_index(ppw->index_ap_ufa,pba->has_ur,index_ap,1);
    class_define_index(ppw->index_ap_ncdmfa,pba->has_ncdm,index_ap,1);

  }

  ppw->ap_size=index_ap;

  if (ppw->ap_size > 0)
    class_alloc(ppw->approx,ppw->ap_size*sizeof(int),ppt->error_message);

  /** - For definiteness, initialize approximation flags to arbitrary
      values (correct values are overwritten in
      pertub_find_approximation_switches) */

  if (_scalars_) {

    ppw->approx[ppw->index_ap_tca]=(int)tca_on;
    ppw->approx[ppw->index_ap_rsa]=(int)rsa_off;
    if (pba->has_ur == _TRUE_) {
      ppw->approx[ppw->index_ap_ufa]=(int)ufa_off;
    }
    if (pba->has_ncdm == _TRUE_) {
      ppw->approx[ppw->index_ap_ncdmfa]=(int)ncdmfa_off;
    }
  }

  if (_tensors_) {

    ppw->approx[ppw->index_ap_tca]=(int)tca_on;
    ppw->approx[ppw->index_ap_rsa]=(int)rsa_off;
  }

  /** - allocate fields where some of the perturbations are stored */

  if (_scalars_) {

    if ((ppt->has_density_transfers == _TRUE_) || (ppt->has_velocity_transfers == _TRUE_) || (ppt->has_source_delta_m == _TRUE_)) {

      class_alloc(ppw->delta_ncdm,pba->N_ncdm*sizeof(double),ppt->error_message);
      class_alloc(ppw->theta_ncdm,pba->N_ncdm*sizeof(double),ppt->error_message);
      class_alloc(ppw->shear_ncdm,pba->N_ncdm*sizeof(double),ppt->error_message);

    }

  }

  return _SUCCESS_;
}

/**
 * Free the perturb_workspace structure (with the exception of the
 * perturb_vector '-->pv' field, which is freed separately in
 * perturb_vector_free).
 *
 * @param ppt        Input: pointer to the perturbation structure
 * @param index_md Input: index of mode under consideration (scalar/.../tensor)
 * @param ppw        Input: pointer to perturb_workspace structure to be freed
 * @return the error status
 */

int perturb_workspace_free (
                            struct perturbs * ppt,
                            int index_md,
                            struct perturb_workspace * ppw
                            ) {

  free(ppw->s_l);
  free(ppw->pvecback);
  free(ppw->pvecthermo);
  free(ppw->pvecmetric);
  if (ppw->ap_size > 0)
    free(ppw->approx);

  if (_scalars_) {

    if ((ppt->has_density_transfers == _TRUE_) || (ppt->has_velocity_transfers == _TRUE_) || (ppt->has_source_delta_m == _TRUE_)) {
      free(ppw->delta_ncdm);
      free(ppw->theta_ncdm);
      free(ppw->shear_ncdm);
    }
  }

  free(ppw);

  return _SUCCESS_;
}

/**
 * Solve the perturbation evolution for a given mode, initial
 * condition and wavenumber, and compute the corresponding source
 * functions.
 *
 * For a given mode, initial condition and wavenumber, this function
 * finds the time ranges over which the perturbations can be described
 * within a given approximation. For each such range, it initializes
 * (or redistributes) perturbations using perturb_vector_init(), and
 * integrates over time. Whenever a "source sampling time" is passed,
 * the source terms are computed and stored in the source table using
 * perturb_sources().
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param ppt        Input/Output: pointer to the perturbation structure (output source functions S(k,tau) written here)
 * @param index_md Input: index of mode under consideration (scalar/.../tensor)
 * @param index_ic   Input: index of initial condition under consideration (ad, iso...)
 * @param index_k    Input: index of wavenumber
 * @param ppw        Input: pointer to perturb_workspace structure containing index values and workspaces
 * @return the error status
 */

int perturb_solve(
                  struct precision * ppr,
                  struct background * pba,
                  struct thermo * pth,
                  struct perturbs * ppt,
                  int index_md,
                  int index_ic,
                  int index_k,
                  struct perturb_workspace * ppw
                  ) {

  /** Summary: */

  /** - define local variables */

  /* contains all fixed parameters, indices and workspaces used by the perturb_derivs function */
  struct perturb_parameters_and_workspace ppaw;

  /* conformal time */
  double tau,tau_lower,tau_upper,tau_mid;

  /* multipole */
  int l;

  /* index running over time */
  int index_tau;

  /* number of values in the tau_sampling array that should be considered for a given mode */
  int tau_actual_size;

  /* running index over types (temperature, etc) */
  int index_type;

  /* Fourier mode */
  double k;

  /* number of time intervals where the approximation scheme is uniform */
  int interval_number;

  /* index running over such time intervals */
  int index_interval;

  /* number of time intervals where each particular approximation is uniform */
  int * interval_number_of;

  /* edge of intervals where approximation scheme is uniform: tau_ini, tau_switch_1, ..., tau_end */
  double * interval_limit;

  /* array of approximation scheme within each interval: interval_approx[index_interval][index_ap] */
  int ** interval_approx;

  /* index running over approximations */
  int index_ap;

  /* approximation scheme within previous interval: previous_approx[index_ap] */
  int * previous_approx;

  int n_ncdm,is_early_enough;

  /* function pointer to ODE evolver and names of possible evolvers */

  extern int evolver_rk();
  extern int evolver_ndf15();
  int (*generic_evolver)();


  /* Related to the perturbation output */
  int (*perhaps_print_variables)();
  int index_ikout;

  /** - initialize indices relevant for back/thermo tables search */
  ppw->last_index_back=0;
  ppw->last_index_thermo=0;
  ppw->inter_mode = pba->inter_normal;

  /** - get wavenumber value */
  k = ppt->k[index_md][index_k];

  class_test(k == 0.,
             ppt->error_message,
             "stop to avoid division by zero");

  /** - If non-zero curvature, update array of free-streaming coefficients ppw->s_l */
  if (pba->has_curvature == _TRUE_){
    for (l = 0; l<=ppw->max_l_max; l++){
      ppw->s_l[l] = sqrt(MAX(1.0-pba->K*(l*l-1.0)/k/k,0.));
    }
  }

  /** - maximum value of tau for which sources are calculated for this wavenumber */

  /* by default, today */
  tau_actual_size = ppt->tau_size;

  /** - using bisection, compute minimum value of tau for which this
      wavenumber is integrated */

  /* will be at least the first time in the background table */
  tau_lower = pba->tau_table[0];

  class_call(background_at_tau(pba,
                               tau_lower,
                               pba->normal_info,
                               pba->inter_normal,
                               &(ppw->last_index_back),
                               ppw->pvecback),
             pba->error_message,
             ppt->error_message);

  class_call(thermodynamics_at_z(pba,
                                 pth,
                                 1./ppw->pvecback[pba->index_bg_a]-1.,
                                 pth->inter_normal,
                                 &(ppw->last_index_thermo),
                                 ppw->pvecback,
                                 ppw->pvecthermo),
             pth->error_message,
             ppt->error_message);

  /* check that this initial time is indeed OK given imposed
     conditions on kappa' and on k/aH */

  class_test(ppw->pvecback[pba->index_bg_a]*
             ppw->pvecback[pba->index_bg_H]/
             ppw->pvecthermo[pth->index_th_dkappa] >
             ppr->start_small_k_at_tau_c_over_tau_h, ppt->error_message, "your choice of initial time for integrating wavenumbers is inappropriate: it corresponds to a time before that at which the background has been integrated. You should increase 'start_small_k_at_tau_c_over_tau_h' up to at least %g, or decrease 'a_ini_over_a_today_default'\n",
             ppw->pvecback[pba->index_bg_a]*
             ppw->pvecback[pba->index_bg_H]/
             ppw->pvecthermo[pth->index_th_dkappa]);

  class_test(k/ppw->pvecback[pba->index_bg_a]/ppw->pvecback[pba->index_bg_H] >
             ppr->start_large_k_at_tau_h_over_tau_k,
             ppt->error_message,
             "your choice of initial time for integrating wavenumbers is inappropriate: it corresponds to a time before that at which the background has been integrated. You should increase 'start_large_k_at_tau_h_over_tau_k' up to at least %g, or decrease 'a_ini_over_a_today_default'\n",
             ppt->k[index_md][ppt->k_size[index_md]-1]/ppw->pvecback[pba->index_bg_a]/ ppw->pvecback[pba->index_bg_H]);

  if (pba->has_ncdm == _TRUE_) {
    for (n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++) {
      class_test(fabs(ppw->pvecback[pba->index_bg_p_ncdm1+n_ncdm]/ppw->pvecback[pba->index_bg_rho_ncdm1+n_ncdm]-1./3.)>ppr->tol_ncdm_initial_w,
                 ppt->error_message,
                 "your choice of initial time for integrating wavenumbers is inappropriate: it corresponds to a time at which the ncdm species number %d is not ultra-relativistic anymore, with w=%g, p=%g and rho=%g\n",
                 n_ncdm,
                 ppw->pvecback[pba->index_bg_p_ncdm1+n_ncdm]/ppw->pvecback[pba->index_bg_rho_ncdm1+n_ncdm],
                 ppw->pvecback[pba->index_bg_p_ncdm1+n_ncdm],
                 ppw->pvecback[pba->index_bg_rho_ncdm1+n_ncdm]);
    }
  }

  /* is at most the time at which sources must be sampled */
  tau_upper = ppt->tau_sampling[0];

  /* start bisection */
  tau_mid = 0.5*(tau_lower + tau_upper);

  while ((tau_upper - tau_lower)/tau_lower > ppr->tol_tau_approx) {

    is_early_enough = _TRUE_;

    class_call(background_at_tau(pba,
                                 tau_mid,
                                 pba->normal_info,
                                 pba->inter_normal,
                                 &(ppw->last_index_back),
                                 ppw->pvecback),
               pba->error_message,
               ppt->error_message);

    /* if there are non-cold relics, check that they are relativistic enough */
    if (pba->has_ncdm == _TRUE_) {
      for (n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++) {
        if (fabs(ppw->pvecback[pba->index_bg_p_ncdm1+n_ncdm]/ppw->pvecback[pba->index_bg_rho_ncdm1+n_ncdm]-1./3.) > ppr->tol_ncdm_initial_w)
          is_early_enough = _FALSE_;
      }
    }

    /* also check that the two conditions on (aH/kappa') and (aH/k) are fulfilled */
    if (is_early_enough == _TRUE_) {

      class_call(thermodynamics_at_z(pba,
                                     pth,
                                     1./ppw->pvecback[pba->index_bg_a]-1.,  /* redshift z=1/a-1 */
                                     pth->inter_normal,
                                     &(ppw->last_index_thermo),
                                     ppw->pvecback,
                                     ppw->pvecthermo),
                 pth->error_message,
                 ppt->error_message);

      if ((ppw->pvecback[pba->index_bg_a]*
           ppw->pvecback[pba->index_bg_H]/
           ppw->pvecthermo[pth->index_th_dkappa] >
           ppr->start_small_k_at_tau_c_over_tau_h) ||
          (k/ppw->pvecback[pba->index_bg_a]/ppw->pvecback[pba->index_bg_H] >
           ppr->start_large_k_at_tau_h_over_tau_k))

        is_early_enough = _FALSE_;
    }

    if (is_early_enough == _TRUE_)
      tau_lower = tau_mid;
    else
      tau_upper = tau_mid;

    tau_mid = 0.5*(tau_lower + tau_upper);

  }

  tau = tau_mid;

  /** - find the number of intervals over which approximation scheme is constant */

  class_alloc(interval_number_of,ppw->ap_size*sizeof(int),ppt->error_message);

  ppw->inter_mode = pba->inter_normal;

  class_call(perturb_find_approximation_number(ppr,
                                               pba,
                                               pth,
                                               ppt,
                                               index_md,
                                               k,
                                               ppw,
                                               tau,
                                               ppt->tau_sampling[tau_actual_size-1],
                                               &interval_number,
                                               interval_number_of),
             ppt->error_message,
             ppt->error_message);

  class_alloc(interval_limit,(interval_number+1)*sizeof(double),ppt->error_message);

  class_alloc(interval_approx,interval_number*sizeof(int*),ppt->error_message);

  for (index_interval=0; index_interval<interval_number; index_interval++)
    class_alloc(interval_approx[index_interval],ppw->ap_size*sizeof(int),ppt->error_message);

  class_call(perturb_find_approximation_switches(ppr,
                                                 pba,
                                                 pth,
                                                 ppt,
                                                 index_md,
                                                 k,
                                                 ppw,
                                                 tau,
                                                 ppt->tau_sampling[tau_actual_size-1],
                                                 ppr->tol_tau_approx,
                                                 interval_number,
                                                 interval_number_of,
                                                 interval_limit,
                                                 interval_approx),
             ppt->error_message,
             ppt->error_message);

  free(interval_number_of);

  /** - fill the structure containing all fixed parameters, indices
      and workspaces needed by perturb_derivs */

  ppaw.ppr = ppr;
  ppaw.pba = pba;
  ppaw.pth = pth;
  ppaw.ppt = ppt;
  ppaw.index_md = index_md;
  ppaw.index_ic = index_ic;
  ppaw.index_k = index_k;
  ppaw.k = k;
  ppaw.ppw = ppw;
  ppaw.ppw->inter_mode = pba->inter_closeby;
  ppaw.ppw->last_index_back = 0;
  ppaw.ppw->last_index_thermo = 0;

  /** - check whether we need to print perturbations to a file for this wavenumber */

  perhaps_print_variables = NULL;
  ppw->index_ikout = -1;
  for (index_ikout=0; index_ikout<ppt->k_output_values_num; index_ikout++){
    if (ppt->index_k_output_values[index_md*ppt->k_output_values_num+index_ikout] == index_k){
      ppw->index_ikout = index_ikout;
      perhaps_print_variables = perturb_print_variables;
      /* class_call(perturb_prepare_output_file(
         pba,ppt,ppw,index_ikout,index_md),
         ppt->error_message,
         ppt->error_message);
      */
    }
  }

  /** - loop over intervals over which approximation scheme is uniform. For each interval: */

  for (index_interval=0; index_interval<interval_number; index_interval++) {

    /** - --> (a) fix the approximation scheme */

    for (index_ap=0; index_ap<ppw->ap_size; index_ap++)
      ppw->approx[index_ap]=interval_approx[index_interval][index_ap];

    /** - --> (b) get the previous approximation scheme. If the current
        interval starts from the initial time tau_ini, the previous
        approximation is set to be a NULL pointer, so that the
        function perturb_vector_init() knows that perturbations must
        be initialized */

    if (index_interval==0) {
      previous_approx=NULL;
    }
    else {
      previous_approx=interval_approx[index_interval-1];
    }

    /** - --> (c) define the vector of perturbations to be integrated
        over. If the current interval starts from the initial time
        tau_ini, fill the vector with initial conditions for each
        mode. If it starts from an approximation switching point,
        redistribute correctly the perturbations from the previous to
        the new vector of perturbations. */

    class_call(perturb_vector_init(ppr,
                                   pba,
                                   pth,
                                   ppt,
                                   index_md,
                                   index_ic,
                                   k,
                                   interval_limit[index_interval],
                                   ppw,
                                   previous_approx),
               ppt->error_message,
               ppt->error_message);

    /** - --> (d) integrate the perturbations over the current interval. */

    if(ppr->evolver == rk){
      generic_evolver = evolver_rk;
    }
    else{
      generic_evolver = evolver_ndf15;
    }

    class_call(generic_evolver(perturb_derivs,
                               interval_limit[index_interval],
                               interval_limit[index_interval+1],
                               ppw->pv->y,
                               ppw->pv->used_in_sources,
                               ppw->pv->pt_size,
                               &ppaw,
                               ppr->tol_perturb_integration,
                               ppr->smallest_allowed_variation,
                               perturb_timescale,
                               ppr->perturb_integration_stepsize,
                               ppt->tau_sampling,
                               tau_actual_size,
                               perturb_sources,
                               perhaps_print_variables,
                               ppt->error_message),
               ppt->error_message,
               ppt->error_message);

  }

  /** - if perturbations were printed in a file, close the file */

  //if (perhaps_print_variables != NULL)
  //  fclose(ppw->perturb_output_file);

  /** - fill the source terms array with zeros for all times between
      the last integrated time tau_max and tau_today. */

  for (index_tau = tau_actual_size; index_tau < ppt->tau_size; index_tau++) {
    for (index_type = 0; index_type < ppt->tp_size[index_md]; index_type++) {
      ppt->sources[index_md]
        [index_ic * ppt->tp_size[index_md] + index_type]
        [index_tau * ppt->k_size[index_md] + index_k] = 0.;
    }
  }

  /** - free quantities allocated at the beginning of the routine */

  class_call(perturb_vector_free(ppw->pv),
             ppt->error_message,
             ppt->error_message);

  for (index_interval=0; index_interval<interval_number; index_interval++)
    free(interval_approx[index_interval]);

  free(interval_approx);

  free(interval_limit);

  return _SUCCESS_;
}

int perturb_prepare_output(struct background * pba,
			   struct perturbs * ppt){

  int n_ncdm;
  char tmp[40];

  ppt->scalar_titles[0]='\0';
  ppt->vector_titles[0]='\0';
  ppt->tensor_titles[0]='\0';


  if (ppt->k_output_values_num > 0) {

    /** Write titles for all perturbations that we would like to print/store. */
    if (ppt->has_scalars == _TRUE_){

      class_store_columntitle(ppt->scalar_titles,"tau [Mpc]",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"a",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"delta_g",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"theta_g",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"shear_g",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"pol0_g",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"pol1_g",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"pol2_g",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"delta_b",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"theta_b",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"psi",_TRUE_);
      class_store_columntitle(ppt->scalar_titles,"phi",_TRUE_);
      /* Perturbed recombination */
      class_store_columntitle(ppt->scalar_titles,"delta_Tb",ppt->has_perturbed_recombination);
      class_store_columntitle(ppt->scalar_titles,"delta_chi",ppt->has_perturbed_recombination);
      /* Ultrarelativistic species */
      class_store_columntitle(ppt->scalar_titles,"delta_ur",pba->has_ur);
      class_store_columntitle(ppt->scalar_titles,"theta_ur",pba->has_ur);
      class_store_columntitle(ppt->scalar_titles,"shear_ur",pba->has_ur);
      /* Cold dark matter */
      class_store_columntitle(ppt->scalar_titles,"delta_cdm",pba->has_cdm);
      class_store_columntitle(ppt->scalar_titles,"theta_cdm",pba->has_cdm);
      /* Non-cold dark matter */
      if ((pba->has_ncdm == _TRUE_) && ((ppt->has_density_transfers == _TRUE_) || (ppt->has_velocity_transfers == _TRUE_) || (ppt->has_source_delta_m == _TRUE_))) {
        for(n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++){
          sprintf(tmp,"delta_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->scalar_titles,tmp,_TRUE_);
          sprintf(tmp,"theta_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->scalar_titles,tmp,_TRUE_);
          sprintf(tmp,"shear_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->scalar_titles,tmp,_TRUE_);
          sprintf(tmp,"cs2_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->scalar_titles,tmp,_TRUE_);
        }
      }
      /* Decaying cold dark matter */
      class_store_columntitle(ppt->scalar_titles, "delta_dcdm", pba->has_dcdm);
      class_store_columntitle(ppt->scalar_titles, "theta_dcdm", pba->has_dcdm);
      /* Decay radiation */
      class_store_columntitle(ppt->scalar_titles, "delta_dr", pba->has_dr);
      class_store_columntitle(ppt->scalar_titles, "theta_dr", pba->has_dr);
      class_store_columntitle(ppt->scalar_titles, "shear_dr", pba->has_dr);
      /* Scalar field scf */
      class_store_columntitle(ppt->scalar_titles, "delta_scf", pba->has_scf);
      class_store_columntitle(ppt->scalar_titles, "theta_scf", pba->has_scf);

      ppt->number_of_scalar_titles =
        get_number_of_titles(ppt->scalar_titles);
    }

    if (ppt->has_tensors == _TRUE_){

      class_store_columntitle(ppt->tensor_titles,"tau [Mpc]",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"a",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"delta_g",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"shear_g",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"l4_g",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"pol0_g",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"pol2_g",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"pol4_g",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"H (gw)",_TRUE_);
      class_store_columntitle(ppt->tensor_titles,"Hdot (gwdot)",_TRUE_);

      class_store_columntitle(ppt->tensor_titles,"delta_ur",ppt->evolve_tensor_ur);
      class_store_columntitle(ppt->tensor_titles,"shear_ur",ppt->evolve_tensor_ur);
      class_store_columntitle(ppt->tensor_titles,"l4_ur",ppt->evolve_tensor_ur);

      if (ppt->evolve_tensor_ncdm == _TRUE_) {
        for(n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++){
          sprintf(tmp,"delta_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->tensor_titles,tmp,_TRUE_);
          sprintf(tmp,"theta_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->tensor_titles,tmp,_TRUE_);
          sprintf(tmp,"shear_ncdm[%d]",n_ncdm);
          class_store_columntitle(ppt->tensor_titles,tmp,_TRUE_);
        }
      }

      ppt->number_of_tensor_titles =
        get_number_of_titles(ppt->tensor_titles);

    }

  }
  return _SUCCESS_;

}


/**
 * For a given mode and wavenumber, find the number of intervals of
 * time between tau_ini and tau_end such that the approximation
 * scheme (and the number of perturbation equations) is uniform.
 *
 * @param ppr                Input: pointer to precision structure
 * @param pba                Input: pointer to background structure
 * @param pth                Input: pointer to the thermodynamics structure
 * @param ppt                Input: pointer to the perturbation structure
 * @param index_md           Input: index of mode under consideration (scalar/.../tensor)
 * @param k                  Input: index of wavenumber
 * @param ppw                Input: pointer to perturb_workspace structure containing index values and workspaces
 * @param tau_ini            Input: initial time of the perturbation integration
 * @param tau_end            Input: final time of the perturbation integration
 * @param interval_number    Output: total number of intervals
 * @param interval_number_of Output: number of intervals with respect to each particular approximation
 * @return the error status
 */

int perturb_find_approximation_number(
                                      struct precision * ppr,
                                      struct background * pba,
                                      struct thermo * pth,
                                      struct perturbs * ppt,
                                      int index_md,
                                      double k,
                                      struct perturb_workspace * ppw,
                                      double tau_ini,
                                      double tau_end,
                                      int * interval_number,
                                      int * interval_number_of /* interval_number_of[index_ap] (already allocated) */
                                      ){

  /** Summary: */
  /* index running over approximations */
  int index_ap;

  /* value of a given approximation at tau_ini and tau_end */
  int flag_ini,flag_end;

  /** - fix default number of intervals to one (if no approximation switch) */

  *interval_number=1;

  /** - loop over each approximation and add the number of approximation switching times */

  for (index_ap=0; index_ap<ppw->ap_size; index_ap++) {

    class_call(perturb_approximations(ppr,
                                      pba,
                                      pth,
                                      ppt,
                                      index_md,
                                      k,
                                      tau_ini,
                                      ppw),
               ppt->error_message,
               ppt->error_message);

    flag_ini = ppw->approx[index_ap];

    class_call(perturb_approximations(ppr,
                                      pba,
                                      pth,
                                      ppt,
                                      index_md,
                                      k,
                                      tau_end,
                                      ppw),
               ppt->error_message,
               ppt->error_message);

    flag_end = ppw->approx[index_ap];

    class_test(flag_end<flag_ini,
               ppt->error_message,
               "For each approximation scheme, the declaration of approximation labels in the enumeration must follow chronological order, e.g: enum approx_flags {flag1, flag2, flag3} with flag1 being the initial one and flag3 the final one");

    *interval_number += flag_end-flag_ini;

    interval_number_of[index_ap] = flag_end-flag_ini+1;
  }

  return _SUCCESS_;

}

/**
 * For a given mode and wavenumber, find the values of time at which
 * the approximation changes.
 *
 * @param ppr                Input: pointer to precision structure
 * @param pba                Input: pointer to background structure
 * @param pth                Input: pointer to the thermodynamics structure
 * @param ppt                Input: pointer to the perturbation structure
 * @param index_md           Input: index of mode under consideration (scalar/.../tensor)
 * @param k                  Input: index of wavenumber
 * @param ppw                Input: pointer to perturb_workspace structure containing index values and workspaces
 * @param tau_ini            Input: initial time of the perturbation integration
 * @param tau_end            Input: final time of the perturbation integration
 * @param precision          Input: tolerance on output values
 * @param interval_number    Input: total number of intervals
 * @param interval_number_of Input: number of intervals with respect to each particular approximation
 * @param interval_limit     Output: value of time at the boundary of the intervals: tau_ini, tau_switch1, ..., tau_end
 * @param interval_approx    Output: value of approximations in each interval
 * @return the error status
 */

int perturb_find_approximation_switches(
                                        struct precision * ppr,
                                        struct background * pba,
                                        struct thermo * pth,
                                        struct perturbs * ppt,
                                        int index_md,
                                        double k,
                                        struct perturb_workspace * ppw,
                                        double tau_ini,
                                        double tau_end,
                                        double precision,
                                        int interval_number,
                                        int * interval_number_of,
                                        double * interval_limit, /* interval_limit[index_interval] (already allocated) */
                                        int ** interval_approx   /* interval_approx[index_interval][index_ap] (already allocated) */
                                        ){

  /** Summary: */

  int index_ap;
  int index_switch;
  int index_switch_tot;
  int num_switch;
  double tau_min,lower_bound,upper_bound;
  double mid=0;
  double * unsorted_tau_switch;
  double next_tau_switch;
  int flag_ini;
  int num_switching_at_given_time;

  /** - write in output arrays the initial time and approximation */

  interval_limit[0]=tau_ini;

  class_call(perturb_approximations(ppr,
                                    pba,
                                    pth,
                                    ppt,
                                    index_md,
                                    k,
                                    tau_ini,
                                    ppw),
             ppt->error_message,
             ppt->error_message);

  for (index_ap=0; index_ap<ppw->ap_size; index_ap++)
    interval_approx[0][index_ap]=ppw->approx[index_ap];

  /** - if there are no approximation switches, just write final time and return */

  if (interval_number == 1) {

    interval_limit[1]=tau_end;

  }

  /** - if there are switches, consider approximations one after each
      other.  Find switching time by bisection. Store all switches in
      arbitrary order in array unsorted_tau_switch[ ] */

  else {

    class_alloc(unsorted_tau_switch,(interval_number-1)*sizeof(double),ppt->error_message);

    index_switch_tot=0;

    for (index_ap=0; index_ap<ppw->ap_size; index_ap++) {

      if (interval_number_of[index_ap] > 1) {

        num_switch = interval_number_of[index_ap]-1;

        tau_min = tau_ini;

        flag_ini = interval_approx[0][index_ap];

        for (index_switch=0; index_switch<num_switch; index_switch++) {

          lower_bound=tau_min;
          upper_bound=tau_end;
          mid = 0.5*(lower_bound+upper_bound);

          while (upper_bound - lower_bound > precision) {

            class_call(perturb_approximations(ppr,
                                              pba,
                                              pth,
                                              ppt,
                                              index_md,
                                              k,
                                              mid,
                                              ppw),
                       ppt->error_message,
                       ppt->error_message);

            if (ppw->approx[index_ap] > flag_ini+index_switch) {
              upper_bound=mid;
            }
            else {
              lower_bound=mid;
            }

            mid = 0.5*(lower_bound+upper_bound);

          }

          unsorted_tau_switch[index_switch_tot]=mid;
          index_switch_tot++;

          tau_min=mid;

        }
      }
    }

    class_test(index_switch_tot != (interval_number-1),
               ppt->error_message,
               "bug in approximation switch search routine: should have %d = %d",
               index_switch_tot,interval_number-1);

    /** - now sort interval limits in correct order */

    index_switch_tot=1;

    while (index_switch_tot < interval_number) {

      next_tau_switch=tau_end;
      for (index_switch=0; index_switch<interval_number-1; index_switch++) {
        if ((unsorted_tau_switch[index_switch] > interval_limit[index_switch_tot-1]) &&
            (unsorted_tau_switch[index_switch] < next_tau_switch)) {
          next_tau_switch=unsorted_tau_switch[index_switch];
        }
      }
      interval_limit[index_switch_tot]=next_tau_switch;
      index_switch_tot++;
    }

    interval_limit[index_switch_tot]=tau_end;

    class_test(index_switch_tot != interval_number,
               ppt->error_message,
               "most probably two approximation switching time were found to be equal, which cannot be handled\n");

    /** - store each approximation in chronological order */

    for (index_switch=1; index_switch<interval_number; index_switch++) {

      class_call(perturb_approximations(ppr,
                                        pba,
                                        pth,
                                        ppt,
                                        index_md,
                                        k,
                                        0.5*(interval_limit[index_switch]+interval_limit[index_switch+1]),
                                        ppw),

                 ppt->error_message,
                 ppt->error_message);

      for (index_ap=0; index_ap<ppw->ap_size; index_ap++) {
        interval_approx[index_switch][index_ap]=ppw->approx[index_ap];

        /* check here that approximation does not go backward (remember
           that by definition the value of an approximation can only
           increase) */
        class_test(interval_approx[index_switch][index_ap] < interval_approx[index_switch-1][index_ap],
                   ppt->error_message,
                   "The approximation with label %d is not defined correctly: it goes backward (from %d to %d) for k=%e and between tau=%e and %e; this cannot be handled\n",
                   index_ap,
                   interval_approx[index_switch-1][index_ap],
                   interval_approx[index_switch][index_ap],
                   k,
                   0.5*(interval_limit[index_switch-1]+interval_limit[index_switch]),
                   0.5*(interval_limit[index_switch]+interval_limit[index_switch+1])
                   );
      }

      /* check here that more than one approximation is not switched on at a given time */
      num_switching_at_given_time=0;
      for (index_ap=0; index_ap<ppw->ap_size; index_ap++) {
        if (interval_approx[index_switch][index_ap] != interval_approx[index_switch-1][index_ap])
          num_switching_at_given_time++;
      }
      class_test(num_switching_at_given_time != 1,
                 ppt->error_message,
                 "for k=%e, at tau=%g, you switch %d approximations at the same time, this cannot be handled. Usually happens in two cases: triggers for different approximations coincide, or one approx is reversible\n",
                 k,
                 interval_limit[index_switch],
                 num_switching_at_given_time);

      if (ppt->perturbations_verbose>2) {

        if (_scalars_) {

          if ((interval_approx[index_switch-1][ppw->index_ap_tca]==(int)tca_on) &&
              (interval_approx[index_switch][ppw->index_ap_tca]==(int)tca_off))
            fprintf(stdout,"Mode k=%e: will switch off tight-coupling approximation at tau=%e\n",k,interval_limit[index_switch]);
          //fprintf(stderr,"Mode k=%e: will switch off tight-coupling approximation at tau=%e\n",k,interval_limit[index_switch]);  //TBC

          if ((interval_approx[index_switch-1][ppw->index_ap_rsa]==(int)rsa_off) &&
              (interval_approx[index_switch][ppw->index_ap_rsa]==(int)rsa_on))
            fprintf(stdout,"Mode k=%e: will switch on radiation streaming approximation at tau=%e\n",k,interval_limit[index_switch]);

          if (pba->has_ur == _TRUE_) {
            if ((interval_approx[index_switch-1][ppw->index_ap_ufa]==(int)ufa_off) &&
                (interval_approx[index_switch][ppw->index_ap_ufa]==(int)ufa_on)) {
              fprintf(stdout,"Mode k=%e: will switch on ur fluid approximation at tau=%e\n",k,interval_limit[index_switch]);
            }
          }
          if (pba->has_ncdm == _TRUE_) {
            if ((interval_approx[index_switch-1][ppw->index_ap_ncdmfa]==(int)ncdmfa_off) &&
                (interval_approx[index_switch][ppw->index_ap_ncdmfa]==(int)ncdmfa_on)) {
              fprintf(stdout,"Mode k=%e: will switch on ncdm fluid approximation at tau=%e\n",k,interval_limit[index_switch]);
            }
          }
        }

        if (_tensors_) {

          if ((interval_approx[index_switch-1][ppw->index_ap_tca]==(int)tca_on) &&
              (interval_approx[index_switch][ppw->index_ap_tca]==(int)tca_off))
            fprintf(stdout,"Mode k=%e: will switch off tight-coupling approximation for tensors at tau=%e\n",k,interval_limit[index_switch]);

          if ((interval_approx[index_switch-1][ppw->index_ap_rsa]==(int)rsa_off) &&
              (interval_approx[index_switch][ppw->index_ap_rsa]==(int)rsa_on))
            fprintf(stdout,"Mode k=%e: will switch on radiation streaming approximation for tensors at tau=%e\n",k,interval_limit[index_switch]);

        }
      }
    }

    free(unsorted_tau_switch);

    class_call(perturb_approximations(ppr,
                                      pba,
                                      pth,
                                      ppt,
                                      index_md,
                                      k,
                                      tau_end,
                                      ppw),

               ppt->error_message,
               ppt->error_message);
  }

  return _SUCCESS_;
}

/**
 * Initialize the field '-->pv' of a perturb_workspace structure, which
 * is a perturb_vector structure. This structure contains indices and
 * values of all quantities which need to be integrated with respect
 * to time (and only them: quantities fixed analytically or obeying
 * constraint equations are NOT included in this vector). This routine
 * distinguishes between two cases:
 *
 * --> the input pa_old is set to the NULL pointer:
 *
 * This happens when we start integrating over a new wavenumber and we
 * want to set initial conditions for the perturbations. Then, it is
 * assumed that ppw-->pv is not yet allocated. This routine allocates
 * it, defines all indices, and then fills the vector ppw-->pv-->y with
 * the initial conditions defined in perturb_initial_conditions.
 *
 * --> the input pa_old is not set to the NULL pointer and describes
 * some set of approximations:
 *
 * This happens when we need to change approximation scheme while
 * integrating over a given wavenumber. The new approximation
 * described by ppw-->pa is then different from pa_old. Then, this
 * routine allocates a new vector with a new size and new index
 * values; it fills this vector with initial conditions taken from the
 * previous vector passed as an input in ppw-->pv, and eventually with
 * some analytic approximations for the new variables appearing at
 * this time; then the new vector comes in replacement of the old one,
 * which is freed.
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param ppt        Input: pointer to the perturbation structure
 * @param index_md Input: index of mode under consideration (scalar/.../tensor)
 * @param index_ic   Input: index of initial condition under consideration (ad, iso...)
 * @param k          Input: wavenumber
 * @param tau        Input: conformal time
 * @param ppw        Input/Output: workspace containing in input the approximation scheme, the background/thermodynamics/metric quantities, and eventually the previous vector y; and in output the new vector y.
 * @param pa_old     Input: NULL is we need to set y to initial conditions for a new wavenumber; points towards a perturb_approximations if we want to switch of approximation.
 * @return the error status
 */

int perturb_vector_init(
                        struct precision * ppr,
                        struct background * pba,
                        struct thermo * pth,
                        struct perturbs * ppt,
                        int index_md,
                        int index_ic,
                        double k,
                        double tau,
                        struct perturb_workspace * ppw, /* ppw->pv unallocated if pa_old = NULL, allocated and filled otherwise */
                        int * pa_old
                        ) {

  /** Summary: */

  /** - define local variables */

  struct perturb_vector * ppv;

  int index_pt;
  int l;
  int n_ncdm,index_q,ncdm_l_size;
  double rho_plus_p_ncdm,q,q2,epsilon,a,factor;

  /** - allocate a new perturb_vector structure to which ppw-->pv will point at the end of the routine */

  class_alloc(ppv,sizeof(struct perturb_vector),ppt->error_message);

  /** - initialize pointers to NULL (they will be allocated later if
      needed), relevant for perturb_vector_free() */
  ppv->l_max_ncdm = NULL;
  ppv->q_size_ncdm = NULL;

  /** - define all indices in this new vector (depends on approximation scheme, described by the input structure ppw-->pa) */

  index_pt = 0;

  if (_scalars_) {

    /* reject inconsistent values of the number of mutipoles in photon temperature hierarchy */
    class_test(ppr->l_max_g < 4,
               ppt->error_message,
               "ppr->l_max_g should be at least 4, i.e. we must integrate at least over photon density, velocity, shear, third and fourth momentum");

    /* reject inconsistent values of the number of mutipoles in photon polarization hierarchy */
    class_test(ppr->l_max_pol_g < 4,
               ppt->error_message,
               "ppr->l_max_pol_g should be at least 4");

    /* reject inconsistent values of the number of mutipoles in decay radiation hierarchy */
    if (pba->has_dr == _TRUE_) {
      class_test(ppr->l_max_dr < 4,
                 ppt->error_message,
                 "ppr->l_max_dr should be at least 4, i.e. we must integrate at least over neutrino/relic density, velocity, shear, third and fourth momentum");
    }

    /* reject inconsistent values of the number of mutipoles in ultra relativistic neutrino hierarchy */
    if (pba->has_ur == _TRUE_) {
      class_test(ppr->l_max_ur < 4,
                 ppt->error_message,
                 "ppr->l_max_ur should be at least 4, i.e. we must integrate at least over neutrino/relic density, velocity, shear, third and fourth momentum");
    }

    /* photons */

    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) { /* if radiation streaming approximation is off */

      /* temperature */

      ppv->l_max_g = ppr->l_max_g;

      class_define_index(ppv->index_pt_delta_g,_TRUE_,index_pt,1); /* photon density */
      class_define_index(ppv->index_pt_theta_g,_TRUE_,index_pt,1); /* photon velocity */

      if (ppw->approx[ppw->index_ap_tca] == (int)tca_off) {

        class_define_index(ppv->index_pt_shear_g,_TRUE_,index_pt,1); /* photon shear */
        class_define_index(ppv->index_pt_l3_g,_TRUE_,index_pt,ppv->l_max_g-2); /* higher momenta */

        /* polarization */

        ppv->l_max_pol_g = ppr->l_max_pol_g;

        class_define_index(ppv->index_pt_pol0_g,_TRUE_,index_pt,1);
        class_define_index(ppv->index_pt_pol1_g,_TRUE_,index_pt,1);
        class_define_index(ppv->index_pt_pol2_g,_TRUE_,index_pt,1);
        class_define_index(ppv->index_pt_pol3_g,_TRUE_,index_pt,ppv->l_max_pol_g-2);
      }
    }

    /* baryons */

    class_define_index(ppv->index_pt_delta_b,_TRUE_,index_pt,1); /* baryon density */
    class_define_index(ppv->index_pt_theta_b,_TRUE_,index_pt,1); /* baryon velocity */

    /* cdm */

    class_define_index(ppv->index_pt_delta_cdm,pba->has_cdm,index_pt,1); /* cdm density */
    class_define_index(ppv->index_pt_theta_cdm,pba->has_cdm && (ppt->gauge == newtonian),index_pt,1); /* cdm velocity */

    /* dcdm */

    class_define_index(ppv->index_pt_delta_dcdm,pba->has_dcdm,index_pt,1); /* dcdm density */
    class_define_index(ppv->index_pt_theta_dcdm,pba->has_dcdm,index_pt,1); /* dcdm velocity */

    /* ultra relativistic decay radiation */
    if (pba->has_dr==_TRUE_){
      ppv->l_max_dr = ppr->l_max_dr;
      class_define_index(ppv->index_pt_F0_dr,_TRUE_,index_pt,ppv->l_max_dr+1); /* all momenta in Boltzmann hierarchy  */
    }

    /* fluid */

    class_define_index(ppv->index_pt_delta_fld,pba->has_fld,index_pt,1); /* fluid density */
    class_define_index(ppv->index_pt_theta_fld,pba->has_fld,index_pt,1); /* fluid velocity */

    /* scalar field */

    class_define_index(ppv->index_pt_phi_scf,pba->has_scf,index_pt,1); /* scalar field density */
    class_define_index(ppv->index_pt_phi_prime_scf,pba->has_scf,index_pt,1); /* scalar field velocity */

    /* perturbed recombination: the indices are defined once tca is off. */
    if ( (ppt->has_perturbed_recombination == _TRUE_) && (ppw->approx[ppw->index_ap_tca] == (int)tca_off) ){
      class_define_index(ppv->index_pt_perturbed_recombination_delta_temp,_TRUE_,index_pt,1);
      class_define_index(ppv->index_pt_perturbed_recombination_delta_chi,_TRUE_,index_pt,1);
    }

    /* ultra relativistic neutrinos */

    if (pba->has_ur && (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off)) {

      class_define_index(ppv->index_pt_delta_ur,_TRUE_,index_pt,1); /* density of ultra-relativistic neutrinos/relics */
      class_define_index(ppv->index_pt_theta_ur,_TRUE_,index_pt,1); /* velocity of ultra-relativistic neutrinos/relics */
      class_define_index(ppv->index_pt_shear_ur,_TRUE_,index_pt,1); /* shear of ultra-relativistic neutrinos/relics */

      if (ppw->approx[ppw->index_ap_ufa] == (int)ufa_off) {
        ppv->l_max_ur = ppr->l_max_ur;
        class_define_index(ppv->index_pt_l3_ur,_TRUE_,index_pt,ppv->l_max_ur-2); /* additional momenta in Boltzmann hierarchy (beyond l=0,1,2,3) */
      }
    }

    /* non-cold dark matter */

    if (pba->has_ncdm == _TRUE_) {
      ppv->index_pt_psi0_ncdm1 = index_pt; /* density of ultra-relativistic neutrinos/relics */
      ppv->N_ncdm = pba->N_ncdm;
      class_alloc(ppv->l_max_ncdm,ppv->N_ncdm*sizeof(double),ppt->error_message);
      class_alloc(ppv->q_size_ncdm,ppv->N_ncdm*sizeof(double),ppt->error_message);

      for(n_ncdm = 0; n_ncdm < pba->N_ncdm; n_ncdm++){
        // Set value of ppv->l_max_ncdm:
        if(ppw->approx[ppw->index_ap_ncdmfa] == (int)ncdmfa_off){
          /* reject inconsistent values of the number of mutipoles in ultra relativistic neutrino hierarchy */
          class_test(ppr->l_max_ncdm < 4,
                     ppt->error_message,
                     "ppr->l_max_ncdm=%d should be at least 4, i.e. we must integrate at least over first four momenta of non-cold dark matter perturbed phase-space distribution",n_ncdm);
          //Copy value from precision parameter:
          ppv->l_max_ncdm[n_ncdm] = ppr->l_max_ncdm;
          ppv->q_size_ncdm[n_ncdm] = pba->q_size_ncdm[n_ncdm];
        }
        else{
          // In the fluid approximation, hierarchy is cut at lmax = 2 and q dependence is integrated out:
          ppv->l_max_ncdm[n_ncdm] = 2;
          ppv->q_size_ncdm[n_ncdm] = 1;
        }
        index_pt += (ppv->l_max_ncdm[n_ncdm]+1)*ppv->q_size_ncdm[n_ncdm];
      }
    }

    /* metric (only quantities to be integrated, not those obeying constraint equations) */

    /* metric perturbation eta of synchronous gauge */
    class_define_index(ppv->index_pt_eta,ppt->gauge == synchronous,index_pt,1);

    /* metric perturbation phi of newtonian gauge ( we could fix it
       using Einstein equations as a constraint equation for phi, but
       integration is numerically more stable if we actually evolve
       phi) */
    class_define_index(ppv->index_pt_phi,ppt->gauge == newtonian,index_pt,1);

  }

  if (_vectors_) {

    /* Vector baryon velocity: v_b^{(1)}. */
    class_define_index(ppv->index_pt_theta_b,_TRUE_,index_pt,1);

    /* eventually reject inconsistent values of the number of mutipoles in photon temperature hierarchy and polarization*/

    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) { /* if radiation streaming approximation is off */
      if (ppw->approx[ppw->index_ap_tca] == (int)tca_off) { /* if tight-coupling approximation is off */

        ppv->l_max_g = ppr->l_max_g_ten;

        class_define_index(ppv->index_pt_delta_g,_TRUE_,index_pt,1); /* photon density */
        class_define_index(ppv->index_pt_theta_g,_TRUE_,index_pt,1); /* photon velocity */
        class_define_index(ppv->index_pt_shear_g,_TRUE_,index_pt,1); /* photon shear */
        class_define_index(ppv->index_pt_l3_g,_TRUE_,index_pt,ppv->l_max_g-2); /* photon l=3 */

        ppv->l_max_pol_g = ppr->l_max_pol_g_ten;

        class_define_index(ppv->index_pt_pol0_g,_TRUE_,index_pt,1); /* photon polarization, l=0 */
        class_define_index(ppv->index_pt_pol1_g,_TRUE_,index_pt,1); /* photon polarization, l=1 */
        class_define_index(ppv->index_pt_pol2_g,_TRUE_,index_pt,1); /* photon polarization, l=2 */
        class_define_index(ppv->index_pt_pol3_g,_TRUE_,index_pt,ppv->l_max_pol_g-2); /* photon polarization, l=3 */
      }
    }

    /** - (a) metric perturbations V or \f$ h_v \f$ depending on gauge */
    if (ppt->gauge == synchronous){
      class_define_index(ppv->index_pt_hv_prime,_TRUE_,index_pt,1);
    }
    if (ppt->gauge == newtonian){
      class_define_index(ppv->index_pt_V,_TRUE_,index_pt,1);
    }

  }

  if (_tensors_) {

    /* reject inconsistent values of the number of mutipoles in photon temperature hierarchy */
    class_test(ppr->l_max_g_ten < 4,
               ppt->error_message,
               "ppr->l_max_g_ten should be at least 4, i.e. we must integrate at least over photon density, velocity, shear, third momentum");

    /* reject inconsistent values of the number of mutipoles in photon polarization hierarchy */
    class_test(ppr->l_max_pol_g_ten < 4,
               ppt->error_message,
               "ppr->l_max_pol_g_ten should be at least 4");

    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) { /* if radiation streaming approximation is off */
      if (ppw->approx[ppw->index_ap_tca] == (int)tca_off) { /* if tight-coupling approximation is off */

        ppv->l_max_g = ppr->l_max_g_ten;

        class_define_index(ppv->index_pt_delta_g,_TRUE_,index_pt,1); /* photon density */
        class_define_index(ppv->index_pt_theta_g,_TRUE_,index_pt,1); /* photon velocity */
        class_define_index(ppv->index_pt_shear_g,_TRUE_,index_pt,1); /* photon shear */
        class_define_index(ppv->index_pt_l3_g,_TRUE_,index_pt,ppv->l_max_g-2); /* photon l=3 */

        ppv->l_max_pol_g = ppr->l_max_pol_g_ten;

        class_define_index(ppv->index_pt_pol0_g,_TRUE_,index_pt,1); /* photon polarization, l=0 */
        class_define_index(ppv->index_pt_pol1_g,_TRUE_,index_pt,1); /* photon polarization, l=1 */
        class_define_index(ppv->index_pt_pol2_g,_TRUE_,index_pt,1); /* photon polarization, l=2 */
        class_define_index(ppv->index_pt_pol3_g,_TRUE_,index_pt,ppv->l_max_pol_g-2); /* photon polarization, l=3 */
      }
    }

    /* ultra relativistic neutrinos */

    class_define_index(ppv->index_pt_delta_ur,ppt->evolve_tensor_ur,index_pt,1); /* ur density  */
    class_define_index(ppv->index_pt_theta_ur,ppt->evolve_tensor_ur,index_pt,1); /* ur velocity */
    class_define_index(ppv->index_pt_shear_ur,ppt->evolve_tensor_ur,index_pt,1); /* ur shear */
    ppv->l_max_ur = ppr->l_max_ur;
    class_define_index(ppv->index_pt_l3_ur,ppt->evolve_tensor_ur,index_pt,ppv->l_max_ur-2); /* additional momenta in Boltzmann hierarchy (beyond l=0,1,2,3) */

    if (ppt->evolve_tensor_ncdm == _TRUE_) {
      ppv->index_pt_psi0_ncdm1 = index_pt;
      ppv->N_ncdm = pba->N_ncdm;
      class_alloc(ppv->l_max_ncdm,ppv->N_ncdm*sizeof(double),ppt->error_message);
      class_alloc(ppv->q_size_ncdm,ppv->N_ncdm*sizeof(double),ppt->error_message);

      for(n_ncdm = 0; n_ncdm < pba->N_ncdm; n_ncdm++){
        // Set value of ppv->l_max_ncdm:
        class_test(ppr->l_max_ncdm < 4,
                   ppt->error_message,
                   "ppr->l_max_ncdm=%d should be at least 4, i.e. we must integrate at least over first four momenta of non-cold dark matter perturbed phase-space distribution",n_ncdm);
        //Copy value from precision parameter:
        ppv->l_max_ncdm[n_ncdm] = ppr->l_max_ncdm;
        ppv->q_size_ncdm[n_ncdm] = pba->q_size_ncdm[n_ncdm];

        index_pt += (ppv->l_max_ncdm[n_ncdm]+1)*ppv->q_size_ncdm[n_ncdm];
      }
    }


    /** - (b) metric perturbation h is a propagating degree of freedom, so h and hdot are included
        in the vector of ordinary perturbations, no in that of metric perturbations */

    class_define_index(ppv->index_pt_gw,_TRUE_,index_pt,1);     /* tensor metric perturbation h (gravitational waves) */
    class_define_index(ppv->index_pt_gwdot,_TRUE_,index_pt,1);  /* its time-derivative */

  }

  ppv->pt_size = index_pt;

  /** - allocate vectors for storing the values of all these
      quantities and their time-derivatives at a given time */

  class_calloc(ppv->y,ppv->pt_size,sizeof(double),ppt->error_message);
  class_alloc(ppv->dy,ppv->pt_size*sizeof(double),ppt->error_message);
  class_alloc(ppv->used_in_sources,ppv->pt_size*sizeof(int),ppt->error_message);

  /** - specify which perturbations are needed in the evaluation of source terms */

  /* take all of them by default */
  for (index_pt=0; index_pt<ppv->pt_size; index_pt++)
    ppv->used_in_sources[index_pt] = _TRUE_;

  /* indicate which ones are not needed (this is just for saving time,
     omitting perturbations in this list will not change the
     results!) */

  if (_scalars_) {

    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {

      if (ppw->approx[ppw->index_ap_tca] == (int)tca_off) {

        /* we don't need temperature multipoles above l=2 (but they are
           defined only when rsa and tca are off) */

        for (index_pt=ppv->index_pt_l3_g; index_pt <= ppv->index_pt_delta_g+ppv->l_max_g; index_pt++)
          ppv->used_in_sources[index_pt]=_FALSE_;

        /* for polarization, we only need l=0,2 (but l =1,3, ... are
           defined only when rsa and tca are off) */

        ppv->used_in_sources[ppv->index_pt_pol1_g]=_FALSE_;

        for (index_pt=ppv->index_pt_pol3_g; index_pt <= ppv->index_pt_pol0_g+ppv->l_max_pol_g; index_pt++)
          ppv->used_in_sources[index_pt]=_FALSE_;

      }

    }

    if (pba->has_ur == _TRUE_) {

      if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {

        if (ppw->approx[ppw->index_ap_ufa] == (int)ufa_off) {

          /* we don't need ur multipoles above l=2 (but they are
             defined only when rsa and ufa are off) */

          for (index_pt=ppv->index_pt_l3_ur; index_pt <= ppv->index_pt_delta_ur+ppv->l_max_ur; index_pt++)
            ppv->used_in_sources[index_pt]=_FALSE_;

        }
      }
    }

    if (pba->has_ncdm == _TRUE_) {

      /* we don't need ncdm multipoles above l=2 (but they are
         defined only when ncdmfa is off) */

      index_pt = ppv->index_pt_psi0_ncdm1;
      for(n_ncdm = 0; n_ncdm < ppv-> N_ncdm; n_ncdm++){
        for(index_q=0; index_q < ppv->q_size_ncdm[n_ncdm]; index_q++){
          for(l=0; l<=ppv->l_max_ncdm[n_ncdm]; l++){
            if (l>2) ppv->used_in_sources[index_pt]=_FALSE_;
            index_pt++;
          }
        }
      }
    }
  }

  if (_tensors_) {

    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) { /* if radiation streaming approximation is off */
      if (ppw->approx[ppw->index_ap_tca] == (int)tca_off) {

        /* we don't need temperature multipoles above except l=0,2,4 */

        ppv->used_in_sources[ppv->index_pt_theta_g]=_FALSE_;
        ppv->used_in_sources[ppv->index_pt_l3_g]=_FALSE_;

        for (index_pt=ppv->index_pt_delta_g+5; index_pt <= ppv->index_pt_delta_g+ppv->l_max_g; index_pt++)
          ppv->used_in_sources[index_pt]=_FALSE_;

        /* same for polarization, we only need l=0,2,4 */

        ppv->used_in_sources[ppv->index_pt_pol1_g]=_FALSE_;
        ppv->used_in_sources[ppv->index_pt_pol3_g]=_FALSE_;

        for (index_pt=ppv->index_pt_pol0_g+5; index_pt <= ppv->index_pt_pol0_g+ppv->l_max_pol_g; index_pt++)
          ppv->used_in_sources[index_pt]=_FALSE_;
      }
    }

    /* we need h' but not h */
    ppv->used_in_sources[ppv->index_pt_gw]=_FALSE_;

  }

  /** - case of setting initial conditions for a new wavenumber */

  if (pa_old == NULL) {

    if (ppt->perturbations_verbose>2)
      fprintf(stdout,"Mode k=%e: initializing vector at tau=%e\n",k,tau);

    if (_scalars_) {

      /** - --> (a) check that current approximation scheme is consistent
          with initial conditions */

      class_test(ppw->approx[ppw->index_ap_rsa] == (int)rsa_on,
                 ppt->error_message,
                 "scalar initial conditions assume radiation streaming approximation turned off");


      if (pba->has_ur == _TRUE_) {

        class_test(ppw->approx[ppw->index_ap_ufa] == (int)ufa_on,
                   ppt->error_message,
                   "scalar initial conditions assume ur fluid approximation turned off");

      }

      if (pba->has_ncdm == _TRUE_) {

        class_test(ppw->approx[ppw->index_ap_ncdmfa] == (int)ncdmfa_on,
                   ppt->error_message,
                   "scalar initial conditions assume ncdm fluid approximation turned off");

      }

      class_test(ppw->approx[ppw->index_ap_tca] == (int)tca_off,
                 ppt->error_message,
                 "scalar initial conditions assume tight-coupling approximation turned on");

    }

    if (_tensors_) {

      class_test(ppw->approx[ppw->index_ap_tca] == (int)tca_off,
                 ppt->error_message,
                 "tensor initial conditions assume tight-coupling approximation turned on");

      class_test(ppw->approx[ppw->index_ap_rsa] == (int)rsa_on,
                 ppt->error_message,
                 "tensor initial conditions assume radiation streaming approximation turned off");

    }

    /** - --> (b) let ppw-->pv points towards the perturb_vector structure
        that we just created */

    ppw->pv = ppv;

    /** - --> (c) fill the vector ppw-->pv-->y with appropriate initial conditions */

    class_call(perturb_initial_conditions(ppr,
                                          pba,
                                          ppt,
                                          index_md,
                                          index_ic,
                                          k,
                                          tau,
                                          ppw),
               ppt->error_message,
               ppt->error_message);

  }

  /** - case of switching approximation while a wavenumber is being integrated */

  else {

    /** - --> (a) for the scalar mode: */

    if (_scalars_) {

      /** - ---> (a.1.) check that the change of approximation scheme makes
          sense (note: before calling this routine there is already a
          check that we wish to change only one approximation flag at
          a time) */

      class_test((pa_old[ppw->index_ap_tca] == (int)tca_off) && (ppw->approx[ppw->index_ap_tca] == (int)tca_on),
                 ppt->error_message,
                 "at tau=%g: the tight-coupling approximation can be switched off, not on",tau);

      /** - ---> (a.2.) some variables (b, cdm, fld, ...) are not affected by
          any approximation. They need to be reconducted whatever
          the approximation switching is. We treat them here. Below
          we will treat other variables case by case. */

      ppv->y[ppv->index_pt_delta_b] =
        ppw->pv->y[ppw->pv->index_pt_delta_b];

      ppv->y[ppv->index_pt_theta_b] =
        ppw->pv->y[ppw->pv->index_pt_theta_b];

      if (pba->has_cdm == _TRUE_) {

        ppv->y[ppv->index_pt_delta_cdm] =
          ppw->pv->y[ppw->pv->index_pt_delta_cdm];

        if (ppt->gauge == newtonian) {
          ppv->y[ppv->index_pt_theta_cdm] =
            ppw->pv->y[ppw->pv->index_pt_theta_cdm];
        }
      }

      if (pba->has_dcdm == _TRUE_) {

        ppv->y[ppv->index_pt_delta_dcdm] =
          ppw->pv->y[ppw->pv->index_pt_delta_dcdm];

        ppv->y[ppv->index_pt_theta_dcdm] =
          ppw->pv->y[ppw->pv->index_pt_theta_dcdm];
      }

      if (pba->has_dr == _TRUE_){
        for (l=0; l <= ppv->l_max_dr; l++)
          ppv->y[ppv->index_pt_F0_dr+l] =
            ppw->pv->y[ppw->pv->index_pt_F0_dr+l];
      }

      if (pba->has_fld == _TRUE_) {

        ppv->y[ppv->index_pt_delta_fld] =
          ppw->pv->y[ppw->pv->index_pt_delta_fld];

        ppv->y[ppv->index_pt_theta_fld] =
          ppw->pv->y[ppw->pv->index_pt_theta_fld];
      }

      if (pba->has_scf == _TRUE_) {

        ppv->y[ppv->index_pt_phi_scf] =
          ppw->pv->y[ppw->pv->index_pt_phi_scf];

        ppv->y[ppv->index_pt_phi_prime_scf] =
          ppw->pv->y[ppw->pv->index_pt_phi_prime_scf];
      }

      if (ppt->gauge == synchronous)
        ppv->y[ppv->index_pt_eta] =
          ppw->pv->y[ppw->pv->index_pt_eta];

      if (ppt->gauge == newtonian)
        ppv->y[ppv->index_pt_phi] =
          ppw->pv->y[ppw->pv->index_pt_phi];

      /* -- case of switching off tight coupling
         approximation. Provide correct initial conditions to new set
         of variables */

      if ((pa_old[ppw->index_ap_tca] == (int)tca_on) && (ppw->approx[ppw->index_ap_tca] == (int)tca_off)) {

        if (ppt->perturbations_verbose>2)
          fprintf(stdout,"Mode k=%e: switch off tight-coupling approximation at tau=%e\n",k,tau);

        ppv->y[ppv->index_pt_delta_g] =
          ppw->pv->y[ppw->pv->index_pt_delta_g];

        ppv->y[ppv->index_pt_theta_g] =
          ppw->pv->y[ppw->pv->index_pt_theta_g];

        /* tight-coupling approximation for shear_g (previously
           computed in perturb_derivs: perturb_derivs is always
           called at the end of generic_evolver, in order to update
           all quantities in ppw to the time at which the
           approximation is switched off) */
        ppv->y[ppv->index_pt_shear_g] = ppw->tca_shear_g;

        ppv->y[ppv->index_pt_l3_g] = 6./7.*k/ppw->pvecthermo[pth->index_th_dkappa]*ppw->s_l[3]*ppv->y[ppv->index_pt_shear_g]; /* second-order tight-coupling approximation for l=3 */

        ppv->y[ppv->index_pt_pol0_g] = 2.5*ppv->y[ppv->index_pt_shear_g];                                                       /* first-order tight-coupling approximation for polarization, l=0 */
        ppv->y[ppv->index_pt_pol1_g] = k/ppw->pvecthermo[pth->index_th_dkappa]*(5.-2.*ppw->s_l[2])/6.*ppv->y[ppv->index_pt_shear_g]; /* second-order tight-coupling approximation for polarization, l=1 */
        ppv->y[ppv->index_pt_pol2_g] = 0.5*ppv->y[ppv->index_pt_shear_g];                                                       /* first-order tight-coupling approximation for polarization, l=2 */
        ppv->y[ppv->index_pt_pol3_g] = k/ppw->pvecthermo[pth->index_th_dkappa]*3.*ppw->s_l[3]/14.*ppv->y[ppv->index_pt_shear_g];     /* second-order tight-coupling approximation for polarization, l=3 */

        if (pba->has_ur == _TRUE_) {

          ppv->y[ppv->index_pt_delta_ur] =
            ppw->pv->y[ppw->pv->index_pt_delta_ur];

          ppv->y[ppv->index_pt_theta_ur] =
            ppw->pv->y[ppw->pv->index_pt_theta_ur];

          ppv->y[ppv->index_pt_shear_ur] =
            ppw->pv->y[ppw->pv->index_pt_shear_ur];

          if (ppw->approx[ppw->index_ap_ufa] == (int)ufa_off) {

            ppv->y[ppv->index_pt_l3_ur] =
              ppw->pv->y[ppw->pv->index_pt_l3_ur];

            for (l=4; l <= ppv->l_max_ur; l++)
              ppv->y[ppv->index_pt_delta_ur+l] =
                ppw->pv->y[ppw->pv->index_pt_delta_ur+l];

          }
        }

        if (pba->has_ncdm == _TRUE_) {
          index_pt = 0;
          for(n_ncdm = 0; n_ncdm < ppv->N_ncdm; n_ncdm++){
            for(index_q=0; index_q < ppv->q_size_ncdm[n_ncdm]; index_q++){
              for(l=0; l<=ppv->l_max_ncdm[n_ncdm];l++){
                // This is correct with or without ncdmfa, since ppv->lmax_ncdm is set accordingly.
                ppv->y[ppv->index_pt_psi0_ncdm1+index_pt] =
                  ppw->pv->y[ppw->pv->index_pt_psi0_ncdm1+index_pt];
                index_pt++;
              }
            }
          }
        }

        /* perturbed recombination */
        /* the initial conditions are set when tca is switched off (current block) */
        if (ppt->has_perturbed_recombination == _TRUE_){
          ppv->y[ppv->index_pt_perturbed_recombination_delta_temp] = 1./3.*ppv->y[ppw->pv->index_pt_delta_b];
          ppv->y[ppv->index_pt_perturbed_recombination_delta_chi] =0.;
        }

      }  // end of block tca ON -> tca OFF

      /* perturbed recombination */
      /* For any other transition in the approximation scheme, we should just copy the value of the perturbations, provided tca is already off (otherwise the indices are not yet allocated). For instance, we do not want to copy the values in the (k,tau) region where both UFA and TCA are engaged.*/

      if ((ppt->has_perturbed_recombination == _TRUE_)&&(pa_old[ppw->index_ap_tca]==(int)tca_off)){
        ppv->y[ppv->index_pt_perturbed_recombination_delta_temp] =
          ppw->pv->y[ppw->pv->index_pt_perturbed_recombination_delta_temp];
        ppv->y[ppv->index_pt_perturbed_recombination_delta_chi] =
          ppw->pv->y[ppw->pv->index_pt_perturbed_recombination_delta_chi];
      }


      /* -- case of switching on radiation streaming
         approximation. Provide correct initial conditions to new set
         of variables */

      if ((pa_old[ppw->index_ap_rsa] == (int)rsa_off) && (ppw->approx[ppw->index_ap_rsa] == (int)rsa_on)) {

        if (ppt->perturbations_verbose>2)
          fprintf(stdout,"Mode k=%e: switch on radiation streaming approximation at tau=%e with Omega_r=%g\n",k,tau,ppw->pvecback[pba->index_bg_Omega_r]);

        if (pba->has_ncdm == _TRUE_) {
          index_pt = 0;
          for(n_ncdm = 0; n_ncdm < ppv->N_ncdm; n_ncdm++){
            for(index_q=0; index_q < ppv->q_size_ncdm[n_ncdm]; index_q++){
              for(l=0; l<=ppv->l_max_ncdm[n_ncdm]; l++){
                ppv->y[ppv->index_pt_psi0_ncdm1+index_pt] =
                  ppw->pv->y[ppw->pv->index_pt_psi0_ncdm1+index_pt];
                index_pt++;
              }
            }
          }
        }
      }

      /* -- case of switching on ur fluid
         approximation. Provide correct initial conditions to new set
         of variables */

      if (pba->has_ur == _TRUE_) {

        if ((pa_old[ppw->index_ap_ufa] == (int)ufa_off) && (ppw->approx[ppw->index_ap_ufa] == (int)ufa_on)) {

          if (ppt->perturbations_verbose>2)
            fprintf(stdout,"Mode k=%e: switch on ur fluid approximation at tau=%e\n",k,tau);

          if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {

            ppv->y[ppv->index_pt_delta_g] =
              ppw->pv->y[ppw->pv->index_pt_delta_g];

            ppv->y[ppv->index_pt_theta_g] =
              ppw->pv->y[ppw->pv->index_pt_theta_g];
          }

          if ((ppw->approx[ppw->index_ap_tca] == (int)tca_off) && (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off)) {

            ppv->y[ppv->index_pt_shear_g] =
              ppw->pv->y[ppw->pv->index_pt_shear_g];

            ppv->y[ppv->index_pt_l3_g] =
              ppw->pv->y[ppw->pv->index_pt_l3_g];

            for (l = 4; l <= ppw->pv->l_max_g; l++) {

              ppv->y[ppv->index_pt_delta_g+l] =
                ppw->pv->y[ppw->pv->index_pt_delta_g+l];
            }

            ppv->y[ppv->index_pt_pol0_g] =
              ppw->pv->y[ppw->pv->index_pt_pol0_g];

            ppv->y[ppv->index_pt_pol1_g] =
              ppw->pv->y[ppw->pv->index_pt_pol1_g];

            ppv->y[ppv->index_pt_pol2_g] =
              ppw->pv->y[ppw->pv->index_pt_pol2_g];

            ppv->y[ppv->index_pt_pol3_g] =
              ppw->pv->y[ppw->pv->index_pt_pol3_g];

            for (l = 4; l <= ppw->pv->l_max_pol_g; l++) {

              ppv->y[ppv->index_pt_pol0_g+l] =
                ppw->pv->y[ppw->pv->index_pt_pol0_g+l];
            }

          }

          if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {

            ppv->y[ppv->index_pt_delta_ur] =
              ppw->pv->y[ppw->pv->index_pt_delta_ur];

            ppv->y[ppv->index_pt_theta_ur] =
              ppw->pv->y[ppw->pv->index_pt_theta_ur];

            ppv->y[ppv->index_pt_shear_ur] =
              ppw->pv->y[ppw->pv->index_pt_shear_ur];
          }

          if (pba->has_ncdm == _TRUE_) {
            index_pt = 0;
            for(n_ncdm = 0; n_ncdm < ppv->N_ncdm; n_ncdm++){
              for(index_q=0; index_q < ppv->q_size_ncdm[n_ncdm]; index_q++){
                for(l=0; l<=ppv->l_max_ncdm[n_ncdm]; l++){
                  /* This is correct even when ncdmfa == off, since ppv->l_max_ncdm and
                      ppv->q_size_ncdm is updated.*/
                  ppv->y[ppv->index_pt_psi0_ncdm1+index_pt] =
                    ppw->pv->y[ppw->pv->index_pt_psi0_ncdm1+index_pt];
                  index_pt++;
                }
              }
            }
          }
        }
      }

      /* -- case of switching on ncdm fluid
         approximation. Provide correct initial conditions to new set
         of variables */

      if (pba->has_ncdm == _TRUE_) {

        if ((pa_old[ppw->index_ap_ncdmfa] == (int)ncdmfa_off) && (ppw->approx[ppw->index_ap_ncdmfa] == (int)ncdmfa_on)) {

          if (ppt->perturbations_verbose>2)
            fprintf(stdout,"Mode k=%e: switch on ncdm fluid approximation at tau=%e\n",k,tau);

          if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {

            ppv->y[ppv->index_pt_delta_g] =
              ppw->pv->y[ppw->pv->index_pt_delta_g];

            ppv->y[ppv->index_pt_theta_g] =
              ppw->pv->y[ppw->pv->index_pt_theta_g];
          }

          if ((ppw->approx[ppw->index_ap_tca] == (int)tca_off) && (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off)) {

            ppv->y[ppv->index_pt_shear_g] =
              ppw->pv->y[ppw->pv->index_pt_shear_g];

            ppv->y[ppv->index_pt_l3_g] =
              ppw->pv->y[ppw->pv->index_pt_l3_g];

            for (l = 4; l <= ppw->pv->l_max_g; l++) {

              ppv->y[ppv->index_pt_delta_g+l] =
                ppw->pv->y[ppw->pv->index_pt_delta_g+l];
            }

            ppv->y[ppv->index_pt_pol0_g] =
              ppw->pv->y[ppw->pv->index_pt_pol0_g];

            ppv->y[ppv->index_pt_pol1_g] =
              ppw->pv->y[ppw->pv->index_pt_pol1_g];

            ppv->y[ppv->index_pt_pol2_g] =
              ppw->pv->y[ppw->pv->index_pt_pol2_g];

            ppv->y[ppv->index_pt_pol3_g] =
              ppw->pv->y[ppw->pv->index_pt_pol3_g];

            for (l = 4; l <= ppw->pv->l_max_pol_g; l++) {

              ppv->y[ppv->index_pt_pol0_g+l] =
                ppw->pv->y[ppw->pv->index_pt_pol0_g+l];
            }

          }

          if (pba->has_ur == _TRUE_) {

            if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {


              ppv->y[ppv->index_pt_delta_ur] =
                ppw->pv->y[ppw->pv->index_pt_delta_ur];

              ppv->y[ppv->index_pt_theta_ur] =
                ppw->pv->y[ppw->pv->index_pt_theta_ur];

              ppv->y[ppv->index_pt_shear_ur] =
                ppw->pv->y[ppw->pv->index_pt_shear_ur];

              if (ppw->approx[ppw->index_ap_ufa] == (int)ufa_off) {

                ppv->y[ppv->index_pt_l3_ur] =
                  ppw->pv->y[ppw->pv->index_pt_l3_ur];

                for (l=4; l <= ppv->l_max_ur; l++)
                  ppv->y[ppv->index_pt_delta_ur+l] =
                    ppw->pv->y[ppw->pv->index_pt_delta_ur+l];

              }
            }
          }

          a = ppw->pvecback[pba->index_bg_a];
          index_pt = ppw->pv->index_pt_psi0_ncdm1;
          for(n_ncdm = 0; n_ncdm < ppv->N_ncdm; n_ncdm++){
            // We are in the fluid approximation, so ncdm_l_size is always 3.
            ncdm_l_size = ppv->l_max_ncdm[n_ncdm]+1;
            rho_plus_p_ncdm = ppw->pvecback[pba->index_bg_rho_ncdm1+n_ncdm]+
              ppw->pvecback[pba->index_bg_p_ncdm1+n_ncdm];
            for(l=0; l<=2; l++){
              ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm+l] = 0.0;
            }
            factor = pba->factor_ncdm[n_ncdm]*pow(pba->a_today/a,4);
            for(index_q=0; index_q < ppw->pv->q_size_ncdm[n_ncdm]; index_q++){
              // Integrate over distributions:
              q = pba->q_ncdm[n_ncdm][index_q];
              q2 = q*q;
              epsilon = sqrt(q2+a*a*pba->M_ncdm[n_ncdm]*pba->M_ncdm[n_ncdm]);
              ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm] +=
                pba->w_ncdm[n_ncdm][index_q]*q2*epsilon*
                ppw->pv->y[index_pt];

              ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm+1] +=
                pba->w_ncdm[n_ncdm][index_q]*q2*q*
                ppw->pv->y[index_pt+1];

              ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm+2] +=
                pba->w_ncdm[n_ncdm][index_q]*q2*q2/epsilon*
                ppw->pv->y[index_pt+2];

              //Jump to next momentum bin in ppw->pv->y:
              index_pt += (ppw->pv->l_max_ncdm[n_ncdm]+1);
            }
            ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm] *=factor/ppw->pvecback[pba->index_bg_rho_ncdm1+n_ncdm];
            ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm+1] *=k*factor/rho_plus_p_ncdm;
            ppv->y[ppv->index_pt_psi0_ncdm1+ncdm_l_size*n_ncdm+2] *=2.0/3.0*factor/rho_plus_p_ncdm;
          }
        }
      }
    }

    /** - --> (b) for the vector mode */

    if (_vectors_) {

      /** - ---> (b.1.) check that the change of approximation scheme makes
          sense (note: before calling this routine there is already a
          check that we wish to change only one approximation flag at
          a time) */

      class_test((pa_old[ppw->index_ap_tca] == (int)tca_off) && (ppw->approx[ppw->index_ap_tca] == (int)tca_on),
                 ppt->error_message,
                 "at tau=%g: the tight-coupling approximation can be switched off, not on",tau);

      /** - ---> (b.2.) some variables (gw, gwdot, ...) are not affected by
          any approximation. They need to be reconducted whatever
          the approximation switching is. We treat them here. Below
          we will treat other variables case by case. */

      if (ppt->gauge == synchronous){

        ppv->y[ppv->index_pt_hv_prime] =
          ppw->pv->y[ppw->pv->index_pt_hv_prime];

      }
      if (ppt->gauge == newtonian){

        ppv->y[ppv->index_pt_V] =
          ppw->pv->y[ppw->pv->index_pt_V];

      }

      ppv->y[ppv->index_pt_theta_b] =
        ppw->pv->y[ppw->pv->index_pt_theta_b];


      /* -- case of switching off tight coupling
         approximation. Provide correct initial conditions to new set
         of variables */

      if ((pa_old[ppw->index_ap_tca] == (int)tca_on) && (ppw->approx[ppw->index_ap_tca] == (int)tca_off)) {

        if (ppt->perturbations_verbose>2)
          fprintf(stdout,"Mode k=%e: switch off tight-coupling approximation at tau=%e\n",k,tau);

        ppv->y[ppv->index_pt_delta_g] = 0.0; //TBC
        //-4./3.*ppw->pv->y[ppw->pv->index_pt_gwdot]/ppw->pvecthermo[pth->index_th_dkappa];

        ppv->y[ppv->index_pt_pol0_g] = 0.0; //TBC
        //1./3.*ppw->pv->y[ppw->pv->index_pt_gwdot]/ppw->pvecthermo[pth->index_th_dkappa];
      }

      /* -- case of switching on radiation streaming
         approximation. Provide correct initial conditions to new set
         of variables */

      if ((pa_old[ppw->index_ap_rsa] == (int)rsa_off) && (ppw->approx[ppw->index_ap_rsa] == (int)rsa_on)) {

        if (ppt->perturbations_verbose>2)
          fprintf(stdout,"Mode k=%e: switch on radiation streaming approximation at tau=%e with Omega_r=%g\n",k,tau,ppw->pvecback[pba->index_bg_Omega_r]);

      }

    }

    /** - --> (c) for the tensor mode */

    if (_tensors_) {

      /** - ---> (c.1.) check that the change of approximation scheme makes
          sense (note: before calling this routine there is already a
          check that we wish to change only one approximation flag at
          a time) */

      class_test((pa_old[ppw->index_ap_tca] == (int)tca_off) && (ppw->approx[ppw->index_ap_tca] == (int)tca_on),
                 ppt->error_message,
                 "at tau=%g: the tight-coupling approximation can be switched off, not on",tau);

      /** - ---> (c.2.) some variables (gw, gwdot, ...) are not affected by
          any approximation. They need to be reconducted whatever
          the approximation switching is. We treat them here. Below
          we will treat other variables case by case. */


      ppv->y[ppv->index_pt_gw] =
        ppw->pv->y[ppw->pv->index_pt_gw];

      ppv->y[ppv->index_pt_gwdot] =
        ppw->pv->y[ppw->pv->index_pt_gwdot];

      if (ppt->evolve_tensor_ur == _TRUE_){

        /* For now, neutrinos go here. */
        ppv->y[ppv->index_pt_delta_ur] =
          ppw->pv->y[ppw->pv->index_pt_delta_ur];

        ppv->y[ppv->index_pt_theta_ur] =
          ppw->pv->y[ppw->pv->index_pt_theta_ur];

        ppv->y[ppv->index_pt_shear_ur] =
          ppw->pv->y[ppw->pv->index_pt_shear_ur];

        ppv->y[ppv->index_pt_l3_ur] =
          ppw->pv->y[ppw->pv->index_pt_l3_ur];

        for (l=4; l <= ppv->l_max_ur; l++)
          ppv->y[ppv->index_pt_delta_ur+l] =
            ppw->pv->y[ppw->pv->index_pt_delta_ur+l];

      }

      if (ppt->evolve_tensor_ncdm == _TRUE_){

        index_pt = 0;
        for(n_ncdm = 0; n_ncdm < ppv->N_ncdm; n_ncdm++){
          for(index_q=0; index_q < ppv->q_size_ncdm[n_ncdm]; index_q++){
            for(l=0; l<=ppv->l_max_ncdm[n_ncdm];l++){
              // This is correct with or without ncdmfa, since ppv->lmax_ncdm is set accordingly.
              ppv->y[ppv->index_pt_psi0_ncdm1+index_pt] =
                ppw->pv->y[ppw->pv->index_pt_psi0_ncdm1+index_pt];
              index_pt++;
            }
          }
        }
      }

      /* -- case of switching off tight coupling
         approximation. Provide correct initial conditions to new set
         of variables */

      if ((pa_old[ppw->index_ap_tca] == (int)tca_on) && (ppw->approx[ppw->index_ap_tca] == (int)tca_off)) {

        if (ppt->perturbations_verbose>2)
          fprintf(stdout,"Mode k=%e: switch off tight-coupling approximation at tau=%e\n",k,tau);

        ppv->y[ppv->index_pt_delta_g] = -4./3.*ppw->pv->y[ppw->pv->index_pt_gwdot]/ppw->pvecthermo[pth->index_th_dkappa];

        ppv->y[ppv->index_pt_pol0_g] = 1./3.*ppw->pv->y[ppw->pv->index_pt_gwdot]/ppw->pvecthermo[pth->index_th_dkappa];
      }

      /* -- case of switching on radiation streaming
         approximation. Provide correct initial conditions to new set
         of variables */

      if ((pa_old[ppw->index_ap_rsa] == (int)rsa_off) && (ppw->approx[ppw->index_ap_rsa] == (int)rsa_on)) {

        if (ppt->perturbations_verbose>2)
          fprintf(stdout,"Mode k=%e: switch on radiation streaming approximation at tau=%e with Omega_r=%g\n",k,tau,ppw->pvecback[pba->index_bg_Omega_r]);

      }
    }

    /** - --> (d) free the previous vector of perturbations */

    class_call(perturb_vector_free(ppw->pv),
               ppt->error_message,
               ppt->error_message);

    /** - --> (e) let ppw-->pv points towards the perturb_vector structure
        that we just created */

    ppw->pv = ppv;

  }

  return _SUCCESS_;
}

/**
 * Free the perturb_vector structure.
 *
 * @param pv        Input: pointer to perturb_vector structure to be freed
 * @return the error status
 */

int perturb_vector_free(
                        struct perturb_vector * pv
                        ) {

  if (pv->l_max_ncdm != NULL) free(pv->l_max_ncdm);
  if (pv->q_size_ncdm != NULL) free(pv->q_size_ncdm);
  free(pv->y);
  free(pv->dy);
  free(pv->used_in_sources);
  free(pv);

  return _SUCCESS_;
}

/**
 * For each mode, wavenumber and initial condition, this function
 * initializes in the vector all values of perturbed variables (in a
 * given gauge). It is assumed here that all values have previously been
 * set to zero, only non-zero values are set here.
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param ppt        Input: pointer to the perturbation structure
 * @param index_md   Input: index of mode under consideration (scalar/.../tensor)
 * @param index_ic   Input: index of initial condition under consideration (ad, iso...)
 * @param k          Input: wavenumber
 * @param tau        Input: conformal time
 * @param ppw        Input/Output: workspace containing in input the approximation scheme, the background/thermodynamics/metric quantities, and eventually the previous vector y; and in output the new vector y.
 * @return the error status
 */

int perturb_initial_conditions(struct precision * ppr,
                               struct background * pba,
                               struct perturbs * ppt,
                               int index_md,
                               int index_ic,
                               double k,
                               double tau,
                               struct perturb_workspace * ppw
                               ) {
  /** Summary: */

  /** --> Declare local variables */

  double a,a_prime_over_a;
  double delta_ur=0.,theta_ur=0.,shear_ur=0.,l3_ur=0.,eta=0.,delta_cdm=0.,alpha, alpha_prime;
  double delta_dr=0;
  double q,epsilon,k2;
  int index_q,n_ncdm,idx;
  double rho_r,rho_m,rho_nu,rho_m_over_rho_r;
  double fracnu,fracg,fracb,fraccdm,om;
  double ktau_two,ktau_three;
  double f_dr;

  double delta_tot;
  double velocity_tot;
  double s2_squared;

  /** --> For scalars */

  if (_scalars_) {

    /** - (a) compute relevant background quantities: compute rho_r,
        rho_m, rho_nu (= all relativistic except photons), and their
        ratio. */

    class_call(background_at_tau(pba,
                                 tau,
                                 pba->normal_info,
                                 pba->inter_normal,
                                 &(ppw->last_index_back),
                                 ppw->pvecback),
               pba->error_message,
               ppt->error_message);

    a = ppw->pvecback[pba->index_bg_a];

    a_prime_over_a = ppw->pvecback[pba->index_bg_H]*a;

    /* 8piG/3 rho_r(t_i) */
    rho_r = ppw->pvecback[pba->index_bg_rho_g];

    /* 8piG/3 rho_m(t_i) */
    rho_m = ppw->pvecback[pba->index_bg_rho_b];

    /* 8piG/3 rho_nu(t_i) (all neutrinos and collisionless relics being relativistic at that time) */
    rho_nu = 0.;

    if (pba->has_cdm == _TRUE_) {
      rho_m += ppw->pvecback[pba->index_bg_rho_cdm];
    }

    if (pba->has_dcdm == _TRUE_) {
      rho_m += ppw->pvecback[pba->index_bg_rho_dcdm];
    }

    if (pba->has_dr == _TRUE_) {
      rho_r += ppw->pvecback[pba->index_bg_rho_dr];
      rho_nu += ppw->pvecback[pba->index_bg_rho_dr];
    }

    if (pba->has_ur == _TRUE_) {
      rho_r += ppw->pvecback[pba->index_bg_rho_ur];
      rho_nu += ppw->pvecback[pba->index_bg_rho_ur];
    }

    if (pba->has_ncdm == _TRUE_) {
      for(n_ncdm=0; n_ncdm<pba->N_ncdm; n_ncdm++){
        rho_r += ppw->pvecback[pba->index_bg_rho_ncdm1 + n_ncdm];
        rho_nu += ppw->pvecback[pba->index_bg_rho_ncdm1 + n_ncdm];
      }
    }

    class_test(rho_r == 0.,
               ppt->error_message,
               "stop to avoid division by zero");

    /* f_nu = Omega_nu(t_i) / Omega_r(t_i) */
    fracnu = rho_nu/rho_r;

    /* f_g = Omega_g(t_i) / Omega_r(t_i) */
    fracg = ppw->pvecback[pba->index_bg_rho_g]/rho_r;

    /* f_b = Omega_b(t_i) / Omega_m(t_i) */
    fracb = ppw->pvecback[pba->index_bg_rho_b]/rho_m;

    /* f_cdm = Omega_cdm(t_i) / Omega_m(t_i) */
    fraccdm = 1.-fracb;

    /* Omega_m(t_i) / Omega_r(t_i) */
    rho_m_over_rho_r = rho_m/rho_r;

    /* omega = Omega_m(t_i) a(t_i) H(t_i) / sqrt(Omega_r(t_i))
       = Omega_m(t_0) a(t_0) H(t_0) / sqrt(Omega_r(t_0)) assuming rho_m in a-3 and rho_r in a^-4
       = (8piG/3 rho_m(t_i)) a(t_i) / sqrt(8piG/3 rho_r(t_i))  in Mpc-1
       This (a priori strange) parameter is the relevant one for expressing a
       as a function of tau during radiation and matter domination (but not DE domination).
       Indeed the exact solution of Friedmann when there is only radiation and matter in
       the universe is
       a = [H(t_0)^2 Omega_m(t_0) a(t_0)^3 / 4] x [tau^2 + 4 tau / omega]
    */
    om = a*rho_m/sqrt(rho_r);

    /* (k tau)^2, (k tau)^3 */
    ktau_two=k*k*tau*tau;
    ktau_three=k*tau*ktau_two;


    /* curvature-dependent factors */

    s2_squared = 1.-3.*pba->K/k/k;

    /** - (b) starts by setting everything in synchronous gauge. If
        another gauge is needed, we will perform a gauge
        transformation below. */

    /** - --> (b.1.) adiabatic */

    if ((ppt->has_ad == _TRUE_) && (index_ic == ppt->index_ic_ad)) {

      /* The following formulas are valid at leading order in
         (k*tau) and (om*tau), and order zero in
         tight-coupling. Identical to first order terms in CRS,
         except for normalization (when ppr->curvature_ini=1, tau=1:
         leads to factor 1/2 difference between CRS formulas with
         beta1=0). Identical to CAMB when om set to zero in theta_g,
         theta_ur, shear_ur, tau

         In the non-flat case the relation R=eta is still valid
         outside the horizon for adiabatic IC. Hence eta is still
         set to ppr->curvature_ini at leading order.  Factors s2
         appear through the solution of Einstein equations and
         equations of motion. */

      /* photon density */
      ppw->pv->y[ppw->pv->index_pt_delta_g] = - ktau_two/3. * (1.-om*tau/5.)
        * ppr->curvature_ini * s2_squared;

      /* photon velocity */
      ppw->pv->y[ppw->pv->index_pt_theta_g] = - k*ktau_three/36. * (1.-3.*(1.+5.*fracb-fracnu)/20./(1.-fracnu)*om*tau)
        * ppr->curvature_ini * s2_squared;

      /* tighly-coupled baryons */
      ppw->pv->y[ppw->pv->index_pt_delta_b] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g]; /* baryon density */
      ppw->pv->y[ppw->pv->index_pt_theta_b] = ppw->pv->y[ppw->pv->index_pt_theta_g]; /* baryon velocity */

      if (pba->has_cdm == _TRUE_) {
        ppw->pv->y[ppw->pv->index_pt_delta_cdm] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g]; /* cdm density */
        /* cdm velocity vanishes in the synchronous gauge */
      }

      if (pba->has_dcdm == _TRUE_) {
        ppw->pv->y[ppw->pv->index_pt_delta_dcdm] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g]; /* dcdm density */
        /* dcdm velocity velocity vanishes initially in the synchronous gauge */

      }


      /* fluid (assumes wa=0, if this is not the case the
         fluid will catch anyway the attractor solution) */
      if (pba->has_fld == _TRUE_) {

        ppw->pv->y[ppw->pv->index_pt_delta_fld] = - ktau_two/4.*(1.+pba->w0_fld+pba->wa_fld)*(4.-3.*pba->cs2_fld)/(4.-6.*(pba->w0_fld+pba->wa_fld)+3.*pba->cs2_fld) * ppr->curvature_ini * s2_squared; /* from 1004.5509 */ //TBC: curvature

        ppw->pv->y[ppw->pv->index_pt_theta_fld] = - k*ktau_three/4.*pba->cs2_fld/(4.-6.*(pba->w0_fld+pba->wa_fld)+3.*pba->cs2_fld) * ppr->curvature_ini * s2_squared; /* from 1004.5509 */ //TBC:curvature

      }

      if (pba->has_scf == _TRUE_) {
        /** - ---> Canonical field (solving for the perturbations):
         *  initial perturbations set to zero, they should reach the attractor soon enough.
         *  - --->  TODO: Incorporate the attractor IC from 1004.5509.
         *  delta_phi \f$ = -(a/k)^2/\phi'(\rho + p)\theta \f$,
         *  delta_phi_prime \f$ = a^2/\phi' \f$ (delta_rho_phi + V'delta_phi),
         *  and assume theta, delta_rho as for perfect fluid
         *  with \f$ c_s^2 = 1 \f$ and w = 1/3 (ASSUMES radiation TRACKING)
        */

        ppw->pv->y[ppw->pv->index_pt_phi_scf] = 0.;
        /*  a*a/k/k/ppw->pvecback[pba->index_bg_phi_prime_scf]*k*ktau_three/4.*1./(4.-6.*(1./3.)+3.*1.) * (ppw->pvecback[pba->index_bg_rho_scf] + ppw->pvecback[pba->index_bg_p_scf])* ppr->curvature_ini * s2_squared; */

        ppw->pv->y[ppw->pv->index_pt_phi_prime_scf] = 0.;
        /* delta_fld expression * rho_scf with the w = 1/3, c_s = 1
            a*a/ppw->pvecback[pba->index_bg_phi_prime_scf]*( - ktau_two/4.*(1.+1./3.)*(4.-3.*1.)/(4.-6.*(1/3.)+3.*1.)*ppw->pvecback[pba->index_bg_rho_scf] - ppw->pvecback[pba->index_bg_dV_scf]*ppw->pv->y[ppw->pv->index_pt_phi_scf])* ppr->curvature_ini * s2_squared; */
      }

      /* all relativistic relics: ur, early ncdm, dr */

      if ((pba->has_ur == _TRUE_) || (pba->has_ncdm == _TRUE_) || (pba->has_dr == _TRUE_)) {

        delta_ur = ppw->pv->y[ppw->pv->index_pt_delta_g]; /* density of ultra-relativistic neutrinos/relics */

        theta_ur = - k*ktau_three/36./(4.*fracnu+15.) * (4.*fracnu+11.+12.*s2_squared-3.*(8.*fracnu*fracnu+50.*fracnu+275.)/20./(2.*fracnu+15.)*tau*om) * ppr->curvature_ini * s2_squared; /* velocity of ultra-relativistic neutrinos/relics */ //TBC

        shear_ur = ktau_two/(45.+12.*fracnu) * (3.*s2_squared-1.) * (1.+(4.*fracnu-5.)/4./(2.*fracnu+15.)*tau*om) * ppr->curvature_ini;//TBC /s2_squared; /* shear of ultra-relativistic neutrinos/relics */  //TBC:0

        l3_ur = ktau_three*2./7./(12.*fracnu+45.)* ppr->curvature_ini;//TBC

        if (pba->has_dr == _TRUE_) delta_dr = delta_ur;

      }

      /* synchronous metric perturbation eta */
      //eta = ppr->curvature_ini * (1.-ktau_two/12./(15.+4.*fracnu)*(5.+4.*fracnu - (16.*fracnu*fracnu+280.*fracnu+325)/10./(2.*fracnu+15.)*tau*om)) /  s2_squared;
      //eta = ppr->curvature_ini * s2_squared * (1.-ktau_two/12./(15.+4.*fracnu)*(15.*s2_squared-10.+4.*s2_squared*fracnu - (16.*fracnu*fracnu+280.*fracnu+325)/10./(2.*fracnu+15.)*tau*om));
      eta = ppr->curvature_ini * (1.-ktau_two/12./(15.+4.*fracnu)*(5.+4.*s2_squared*fracnu - (16.*fracnu*fracnu+280.*fracnu+325)/10./(2.*fracnu+15.)*tau*om));

    }

    /* isocurvature initial conditions taken from Bucher, Moodely,
       Turok 99, with just a different normalization convention for
       tau and the scale factor. [k tau] from BMT99 is left invariant
       because it is the ratio [k/aH]. But [Omega_i,0 tau] from BMT99
       must be replaced by [frac_i*om*tau/4]. Some doubts remain about
       the niv formulas, that should be recheked at some point. We
       also checked that for bi,cdi,nid, everything coincides exactly
       with the CAMB formulas. */

    /** - --> (b.2.) Cold dark matter Isocurvature */

    if ((ppt->has_cdi == _TRUE_) && (index_ic == ppt->index_ic_cdi)) {

      class_test(pba->has_cdm == _FALSE_,
                 ppt->error_message,
                 "not consistent to ask for CDI in absence of CDM!");

      ppw->pv->y[ppw->pv->index_pt_delta_g] = ppr->entropy_ini*fraccdm*om*tau*(-2./3.+om*tau/4.);
      ppw->pv->y[ppw->pv->index_pt_theta_g] = -ppr->entropy_ini*fraccdm*om*ktau_two/12.;

      ppw->pv->y[ppw->pv->index_pt_delta_b] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g];
      ppw->pv->y[ppw->pv->index_pt_theta_b] = ppw->pv->y[ppw->pv->index_pt_theta_g];

      ppw->pv->y[ppw->pv->index_pt_delta_cdm] = ppr->entropy_ini+3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g];

      if ((pba->has_ur == _TRUE_) || (pba->has_ncdm == _TRUE_)) {

        delta_ur = ppw->pv->y[ppw->pv->index_pt_delta_g];
        theta_ur = ppw->pv->y[ppw->pv->index_pt_theta_g];
        shear_ur = -ppr->entropy_ini*fraccdm*ktau_two*tau*om/6./(2.*fracnu+15.);

      }

      eta = -ppr->entropy_ini*fraccdm*om*tau*(1./6.-om*tau/16.);

    }

    /** - --> (b.3.) Baryon Isocurvature */

    if ((ppt->has_bi == _TRUE_) && (index_ic == ppt->index_ic_bi)) {

      ppw->pv->y[ppw->pv->index_pt_delta_g] = ppr->entropy_ini*fracb*om*tau*(-2./3.+om*tau/4.);
      ppw->pv->y[ppw->pv->index_pt_theta_g] = -ppr->entropy_ini*fracb*om*ktau_two/12.;

      ppw->pv->y[ppw->pv->index_pt_delta_b] = ppr->entropy_ini+3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g];
      ppw->pv->y[ppw->pv->index_pt_theta_b] = ppw->pv->y[ppw->pv->index_pt_theta_g];

      if (pba->has_cdm == _TRUE_) {

        ppw->pv->y[ppw->pv->index_pt_delta_cdm] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g];

      }

      if ((pba->has_ur == _TRUE_) || (pba->has_ncdm == _TRUE_)) {

        delta_ur = ppw->pv->y[ppw->pv->index_pt_delta_g];
        theta_ur = ppw->pv->y[ppw->pv->index_pt_theta_g];
        shear_ur = -ppr->entropy_ini*fracb*ktau_two*tau*om/6./(2.*fracnu+15.);

      }

      eta = -ppr->entropy_ini*fracb*om*tau*(1./6.-om*tau/16.);

    }

    /** - --> (b.4.) Neutrino density Isocurvature */

    if ((ppt->has_nid == _TRUE_) && (index_ic == ppt->index_ic_nid)) {

      class_test((pba->has_ur == _FALSE_) && (pba->has_ncdm == _FALSE_),
                 ppt->error_message,
                 "not consistent to ask for NID in absence of ur or ncdm species!");

      ppw->pv->y[ppw->pv->index_pt_delta_g] = ppr->entropy_ini*fracnu/fracg*(-1.+ktau_two/6.);
      ppw->pv->y[ppw->pv->index_pt_theta_g] = -ppr->entropy_ini*fracnu/fracg*k*k*tau*(1./4.-fracb/fracg*3./16.*om*tau);

      ppw->pv->y[ppw->pv->index_pt_delta_b] = ppr->entropy_i