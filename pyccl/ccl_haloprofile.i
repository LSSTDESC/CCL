%module ccl_haloprofile

#include "../include/ccl_haloprofile.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* r, int nr),
     (double *rs, int nrs),
     (double *rd, int nrd),
     (double *al, int nal)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") einasto_norm %{
    if numpy.shape(rs) != numpy.shape(rd):
        raise CCLError("Input shape for `rs` must match `rd`!")
    if numpy.shape(rs) != numpy.shape(al):
        raise CCLError("Input shape for `rs` must match `al`!")
    if numpy.shape(rs) != (nout,):
        raise CCLError("Input shape for `rs` must match `(nout,)`!")
%}
%inline %{
void einasto_norm(double *rs,int nrs,
		  double *rd,int nrd,
		  double *al,int nal,
		  int nout,double *output,
		  int *status)
{
  ccl_einasto_norm_integral(nrs,rs,rd,al,output,status);
}
%}

%feature("pythonprepend") hernquist_norm %{
    if numpy.shape(rs) != numpy.shape(rd):
        raise CCLError("Input shape for `rs` must match `rd`!")
    if numpy.shape(rs) != (nout,):
        raise CCLError("Input shape for `rs` must match `(nout,)`!")
%}
%inline %{
void hernquist_norm(double *rs,int nrs,
		    double *rd,int nrd,
		    int nout,double *output,
		    int *status)
{
  ccl_hernquist_norm_integral(nrs,rs,rd,output,status);
}
%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
