%module ccl_utils

%{
/* put additional #include here */
%}

%include "../include/ccl_utils.h"

%apply (double* IN_ARRAY1, int DIM1) {
  (double *x_in, int n_in_x),
  (double *ys_in, int n_in_y)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%apply (int DIM1, double* ARGOUT_ARRAY1) {(int x_size, double* xarr)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int y_size, double* yarr)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int z_size, double* zarr)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int t_size, double* tarr)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int a_size, double* out_arr)};

%apply (int* OUTPUT) {(int *size)};
%apply int *OUTPUT { int *x_size, int *y_size };


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

void get_spline1d_array_size(gsl_spline *spline, int* size, int* status) {
  if(spline == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  if(spline->interp == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  *size = spline->interp->size;
}

void get_spline2d_array_sizes(gsl_spline2d *spline2d, int* x_size, int* y_size,
                              int* status) {
  if(spline2d == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  *x_size = spline2d->interp_object.xsize;
  *y_size = spline2d->interp_object.ysize;
}

// Not really 3d, because 3d is an array of 2d interps, but works with double pointers. //
void get_spline3d_array_sizes(gsl_spline2d **spline2d, int* x_size, int* y_size,
                              int* status) {
  if((*spline2d) == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  // check only the first item of the array //
  *x_size = (*spline2d)->interp_object.xsize;
  *y_size = (*spline2d)->interp_object.ysize;
}

void get_spline1d_arrays(gsl_spline *spline,
                         int x_size, double* xarr,
                         int y_size, double* yarr,
                         int *status)
{
  if(spline == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  if(x_size != spline->interp->size) {
    *status = CCL_ERROR_INCONSISTENT;
    return;
  }
  if(y_size != spline->interp->size) {
    *status = CCL_ERROR_INCONSISTENT;
    return;
  }
  memcpy(xarr, spline->x, sizeof(double)*x_size);
  memcpy(yarr, spline->y, sizeof(double)*y_size);
}

void get_spline2d_arrays(gsl_spline2d *spline2d,
                         int x_size, double* xarr,
                         int y_size, double* yarr,
                         int z_size, double* zarr,
                         int *status)
{
  if(spline2d == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  if(x_size != spline2d->interp_object.xsize) {
    *status = CCL_ERROR_INCONSISTENT;
    return;
  }
  if(y_size != spline2d->interp_object.ysize) {
    *status = CCL_ERROR_INCONSISTENT;
    return;
  }
  memcpy(xarr, spline2d->xarr, sizeof(double)*x_size);
  memcpy(yarr, spline2d->yarr, sizeof(double)*y_size);
  memcpy(zarr, spline2d->zarr, sizeof(double)*x_size*y_size);
}

void get_spline3d_arrays(gsl_spline2d **spline2d,
                         int x_size, double* xarr,
                         int y_size, double* yarr,
                         int t_size, double* tarr,
                         int na,
                         int *status)
{
  // check for inconsistencies //
  if (spline2d == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }

  for (int ia = 0; ia < na; ia++) {
    if (x_size != spline2d[ia]->interp_object.xsize
        || y_size != spline2d[ia]->interp_object.ysize) {
      *status = CCL_ERROR_INCONSISTENT;
      return;
    }
  }

  for (int ia = 0; ia < na; ia++) {
    for (int ik = 0; ik < x_size*y_size; ik++) {
      tarr[ia*x_size*y_size + ik] = spline2d[ia]->zarr[ik];
    }
  }

  // no need to do this for every scale factor //
  memcpy(xarr, spline2d[0]->xarr, sizeof(double)*x_size);
  memcpy(yarr, spline2d[0]->yarr, sizeof(double)*y_size);
}


void get_array(double *arr,
               int a_size, double* out_arr,
               int *status)
{
  if(arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  memcpy(out_arr, arr, sizeof(double)*a_size);
}

%}
