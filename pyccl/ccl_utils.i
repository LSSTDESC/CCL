%module ccl_utils

%{
/* put additional #include here */
%}

%include "../include/ccl_utils.h"

%apply (double* IN_ARRAY1, int DIM1) {
  (double *x_in, int n_in_x),
  (double *ys_in, int n_in_y)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") spline_integrate %{
    if n_integ * x_in.size != ys_in.size:
        raise CCLError("Input size for `ys_in` must match `n_integ * x_in.size`")

    if nout != n_integ:
        raise CCLError("Input shape for `output` must match n_integ")
%}

%inline %{


void spline_integrate(int n_integ,
                      double *x_in, int n_in_x,
                      double *ys_in, int n_in_y,
                      double a, double b,
                      int nout, double *output,
                      int *status)
{
  int ii;

  double **_ys_in=NULL;
  _ys_in = malloc(n_integ*sizeof(double *));
  if(_ys_in==NULL)
    *status = CCL_ERROR_MEMORY;

  if(*status==0) {
    for(ii=0;ii<n_integ;ii++)
      _ys_in[ii]=&(ys_in[ii*n_in_x]);

    ccl_integ_spline(n_integ, n_in_x, x_in, _ys_in,
                     a, b, output, gsl_interp_akima,
                     status);
  }

  free(_ys_in);
}

%}
