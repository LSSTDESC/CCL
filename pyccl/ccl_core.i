%module ccl_core

%{
/* put additional #include here */
%}

// SWIG black magic. Change the behaviour of setting A_SPLINE_MAX to throwing an
// error.
%typemap(memberin) double A_SPLINE_MAX {
    if($input) {
        PyErr_SetString(PyExc_RuntimeError, "A_SPLINE_MAX is fixed to 1.0 and is not mutable.");
        SWIG_fail;
    }
}

%include "../include/ccl_core.h"


%apply (double* IN_ARRAY1, int DIM1) {
       (double* mass, int num),
       (double* zarr, int nz),
       (double* dfarr, int ndf)
};

%inline %{
void parameters_m_nu_set_custom(ccl_parameters *params, double *mass, int num) {
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
    output[0] = 0;
    output[1] = 0;
    output[2] = 0;

    for (int i=0; i<params->N_nu_mass; ++i) {
        output[i] = params->m_nu[i];
    }
}

%}
