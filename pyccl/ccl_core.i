%module ccl_core

%{
/* put additional #include here */
%}

%include "../include/ccl_core.h"


%apply (double* IN_ARRAY1, int DIM1) {
       (double* mass, int num),
       (double* zarr, int nz),
       (double* dfarr, int ndf)
};

%inline %{
void parameters_m_nu_set_custom(ccl_parameters *params, double *mass, int num) {
  if(params->m_nu != NULL) {
    free(params->m_nu);
  }
  params->m_nu = (double*) malloc(num*sizeof(double));
  memcpy(params->m_nu, mass, num*sizeof(double));
}

void parameters_mgrowth_set_custom(ccl_parameters *params,
    double* zarr, int nz, double* dfarr, int ndf) {
  if (nz > 0) {
    params->has_mgrowth = true;
    params->nz_mgrowth = nz;
    params->z_mgrowth = (double*) malloc(nz*sizeof(double));
    params->df_mgrowth = (double*) malloc(nz*sizeof(double));
    memcpy(params->z_mgrowth, zarr, nz*sizeof(double));
    memcpy(params->df_mgrowth, dfarr, nz*sizeof(double));
  }
}

%}


%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%inline %{

void parameters_get_nu_masses(ccl_parameters *params, int nout, double* output) {
  memcpy(output, params->m_nu, nout*sizeof(double));
}

%}
