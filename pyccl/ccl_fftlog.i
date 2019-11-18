%module ccl_f1d

%{
/* put additional #include here */
%}

%include "../include/ccl_fftlog.h"

%apply (double* IN_ARRAY1, int DIM1) {
  (double *k_in, int n_in_k),
  (double *fk_in, int n_in_f)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") fftlog_transform %{
    if numpy.shape(k_in) != numpy.shape(fk_in):
        raise CCLError("Input shape for `k_in` must match `fk_in`!")

    if nout != 2 * k_in.size:
        raise CCLError("Input shape for `output` must match `(2 * k_in.size,)`!")

    if (dim != 2) and (dim !=3):
        raise CCLError("`dim` must be 2 or 3")
%}

%inline %{


void fftlog_transform(double *k_in, int n_in_k,
		      double *fk_in, int n_in_f,
		      int dim,
		      int nout, double *output,
		      int *status)
{
  double *r_out = &(output[0]);
  double *fr_out = &(output[n_in_k]);
  if(dim==3)
    ccl_fftlog_ComputeXi3D(0, 0, n_in_k, k_in, fk_in, r_out, fr_out);
  else if(dim==2)
    ccl_fftlog_ComputeXi2D(0, 0, n_in_k, k_in, fk_in, r_out, fr_out);
}

%}
