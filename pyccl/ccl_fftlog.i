%module ccl_fftlog

%{
/* put additional #include here */
%}

%include "../include/ccl_fftlog.h"

%apply (double* IN_ARRAY1, int DIM1) {
  (double *k_in, int n_in_k),
  (double *fk_in, int n_in_f)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") fftlog_transform %{
    if npk * k_in.size != fk_in.size:
        raise CCLError("Input size for `fk_in` must match `npk * k_in.size`")

    if nout != k_in.size * (npk + 1):
        raise CCLError("Input shape for `output` must match `(k_in.size * (npk + 1),)`!")

    if (dim != 2) and (dim !=3):
        raise CCLError("`dim` must be 2 or 3")
%}


%inline %{


void fftlog_transform(int npk,
          double *k_in, int n_in_k,
          double *fk_in, int n_in_f,
          int dim, double mu, double plaw_index,
          int nout, double *output,
          int *status)
{
  int ii;
  double *r_out = &(output[0]);
  double epsilon = 0.5*dim+plaw_index;

  double **_fk_in=NULL, **_fr_out=NULL;
  _fk_in = malloc(npk*sizeof(double *));
  _fr_out = malloc(npk*sizeof(double *));
  if((_fk_in==NULL) || (_fr_out==NULL))
    *status = CCL_ERROR_MEMORY;

  if(*status==0) {
    for(ii=0;ii<npk;ii++) {
      _fk_in[ii]=&(fk_in[ii*n_in_k]);
      _fr_out[ii]=&(output[(ii+1)*n_in_k]);
    }

    if(dim==3)
      ccl_fftlog_ComputeXi3D(mu, epsilon, npk, n_in_k, k_in, _fk_in, r_out, _fr_out, status);
    if(dim==2)
      ccl_fftlog_ComputeXi2D(mu, epsilon, npk, n_in_k, k_in, _fk_in, r_out, _fr_out, status);
  }

  free(_fk_in);
  free(_fr_out);
}

%}

%feature("pythonprepend") fftlog_transform_general %{
    if npk * k_in.size != fk_in.size:
        raise CCLError("Input size for `fk_in` must match `npk * k_in.size`")

    if nout != k_in.size * (npk + 1):
        raise CCLError("Input shape for `output` must match `(k_in.size * (npk + 1),)`!")

    if round(bessel_deriv)!= bessel_deriv or bessel_deriv<0:
        raise CCLError("bessel_deriv must be a non-negative integer in a floating-point format.`")
    if spherical_bessel!=0 and spherical_bessel!=1:
        raise CCLError("0 and 1 are the only valid values of spherical_bessel.")
%}


%inline %{
void fftlog_transform_general(int npk,
          double *k_in, int n_in_k,
          double *fk_in, int n_in_f,
          double mu, double q,
          int spherical_bessel, double bessel_deriv, double plaw,
          int nout, double *output,
          int *status)
{
  int ii;
  double *r_out = &(output[0]);

  double **_fk_in=NULL, **_fr_out=NULL;
  _fk_in = malloc(npk*sizeof(double *));
  _fr_out = malloc(npk*sizeof(double *));
  if((_fk_in==NULL) || (_fr_out==NULL))
    *status = CCL_ERROR_MEMORY;

  if(*status==0) {
    for(ii=0;ii<npk;ii++) {
      _fk_in[ii]=&(fk_in[ii*n_in_k]);
      _fr_out[ii]=&(output[(ii+1)*n_in_k]);
    }
    ccl_fftlog_ComputeXi_general(mu, q, npk, n_in_k, k_in, _fk_in, spherical_bessel, bessel_deriv, plaw, r_out, _fr_out, status);
  }

  free(_fk_in);
  free(_fr_out);
}

%}
