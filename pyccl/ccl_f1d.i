%module ccl_f1d

%{
/* put additional #include here */
%}

%include "../include/ccl_f1d.h"

%apply (double* IN_ARRAY1, int DIM1) {
  (double *x_in, int n_in_x),
  (double *f_in, int n_in_f),
  (double *x_out, int n_out_x)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") array_1d_resample %{
    if numpy.shape(x_in) != numpy.shape(f_in):
        raise CCLError("Input shape for `x_in` must match `f_in`!")

    if numpy.shape(x_out) != (nout,):
        raise CCLError("Input shape for `x_out` must match `(nout,)`!")
%}

%inline %{

void array_1d_resample(double *x_in, int n_in_x,
		       double *f_in, int n_in_f,
		       double *x_out, int n_out_x,
		       double f0, double ff,
		       int extrap_lo, int extrap_hi,
		       int nout, double *output,
		       int *status)
{
  int ii;
  ccl_f1d_t *spl=ccl_f1d_t_new(n_in_x, x_in, f_in, f0, ff,
			       extrap_hi, extrap_lo, status);
  if(spl==NULL)
    *status=CCL_ERROR_MEMORY;

  if(*status==0) {
    for(ii=0;ii<n_out_x;ii++) {
      double ret = ccl_f1d_t_eval(spl, x_out[ii]);
      if(ret!=ret) { //Check for NAN
	*status=CCL_ERROR_SPLINE_EV;
	break;
      }
      output[ii] = ret;
    }
  }

  ccl_f1d_t_free(spl);
}

%}
