/**
 * module with tools for manipulating arrays
 * Julien Lesgourgues, 18.04.2010
 */

#include "arrays.h"

/**
 * Called by thermodynamics_init(); perturb_sources().
 */
int array_derive(
		 double * array,
		 int n_columns,
		 int n_lines,
		 int index_x,   /** from 0 to (n_columns-1) */
		 int index_y,
		 int index_dydx,
		 ErrorMsg errmsg) {

  int i;

  double dx1,dx2,dy1,dy2,weight1,weight2;

  class_test((index_dydx == index_x) || (index_dydx == index_y),
	     errmsg,
	     "output column %d must differ from input columns %d and %d",index_dydx,index_x,index_y);

  dx2=array[1*n_columns+index_x]-array[0*n_columns+index_x];
  dy2=array[1*n_columns+index_y]-array[0*n_columns+index_y];

  for (i=1; i<n_lines-1; i++) {

    dx1 = dx2;
    dy1 = dy2;
    dx2 = array[(i+1)*n_columns+index_x]-array[i*n_columns+index_x];
    dy2 = array[(i+1)*n_columns+index_y]-array[i*n_columns+index_y];
    class_test((dx1 == 0) || (dx2 == 0),
	       errmsg,
	       "stop to avoid division by zero");
    weight1 = dx2*dx2;
    weight2 = dx1*dx1;
    array[i*n_columns+index_dydx] = (weight1*dy1+weight2*dy2) / (weight1*dx1+weight2*dx2);

    if (i == 1)
      array[(i-1)*n_columns+index_dydx] = 2.*dy1/dx1 - array[i*n_columns+index_dydx];

    if (i == n_lines-2)
      array[(i+1)*n_columns+index_dydx] = 2.*dy2/dx2 - array[i*n_columns+index_dydx];
  }

  return _SUCCESS_;
}

int array_derive_spline(
		 double * x_array,
		 int n_lines,
		 double * array,
		 double * array_splined,
		 int n_columns,
		 int index_y,
		 int index_dydx,
		 ErrorMsg errmsg) {

  int i;

  double h;

  class_test(index_dydx == index_y,
	     errmsg,
	     "Output column %d must differ from input columns %d",
	     index_dydx,
	     index_y);

  class_test(n_lines<2,
	     errmsg,
	     "no possible derivation with less than two lines");

  for (i=0; i<n_lines-1; i++) {

    h = x_array[i+1] - x_array[i];
    if (h == 0) {
      sprintf(errmsg,"%s(L:%d) h=0, stop to avoid division by zero",__func__,__LINE__);
      return _FAILURE_;
    }

    array[i*n_columns+index_dydx] =
      (array[(i+1)*n_columns+index_y] - array[i*n_columns+index_y])/h
      - h / 6. * (array_splined[(i+1)*n_columns+index_y] + 2. * array_splined[i*n_columns+index_y]);

  }

  h = x_array[n_lines-1] - x_array[n_lines-2];

  array[(n_lines-1)*n_columns+index_dydx] =
    (array[(n_lines-1)*n_columns+index_y] - array[(n_lines-2)*n_columns+index_y])/h
    + h / 6. * (2. * array_splined[(n_lines-1)*n_columns+index_y] + array_splined[(n_lines-2)*n_columns+index_y]);

  return _SUCCESS_;
}

int array_derive_spline_table_line_to_line(
					   double * x_array,
					   int n_lines,
					   double * array,
					   int n_columns,
					   int index_y,
					   int index_ddy,
					   int index_dy,
					   ErrorMsg errmsg) {

  int i;

  double h;

  class_test(index_ddy == index_y,
	     errmsg,
	     "Output column %d must differ from input columns %d",
	     index_ddy,
	     index_y);

  class_test(index_ddy == index_dy,
	     errmsg,
	     "Output column %d must differ from input columns %d",
	     index_ddy,
	     index_dy);

  class_test(n_lines<2,
	     errmsg,
	     "no possible derivation with less than two lines");

  for (i=0; i<n_lines-1; i++) {

    h = x_array[i+1] - x_array[i];
    if (h == 0) {
      sprintf(errmsg,"%s(L:%d) h=0, stop to avoid division by zero",__func__,__LINE__);
      return _FAILURE_;
    }

    array[i*n_columns+index_dy] =
      (array[(i+1)*n_columns+index_y] - array[i*n_columns+index_y])/h
      - h / 6. * (array[(i+1)*n_columns+index_ddy] + 2. * array[i*n_columns+index_ddy]);

  }

  h = x_array[n_lines-1] - x_array[n_lines-2];

  array[(n_lines-1)*n_columns+index_dy] =
    (array[(n_lines-1)*n_columns+index_y] - array[(n_lines-2)*n_columns+index_y])/h
    + h / 6. * (2. * array[(n_lines-1)*n_columns+index_ddy] + array[(n_lines-2)*n_columns+index_ddy]);

  return _SUCCESS_;
}

int array_derive1_order2_table_line_to_line(
				       double * x_array,
				       int n_lines,
				       double * array,
				       int n_columns,
				       int index_y,
				       int index_dy,
				       ErrorMsg errmsg) {

  int i=1;
  double dxp,dxm,dyp,dym;

  if (n_lines < 2) {
    sprintf(errmsg,"%s(L:%d) routine called with n_lines=%d, should be at least 2",__func__,__LINE__,n_lines);
    return _FAILURE_;
  }

  dxp = x_array[2] - x_array[1];
  dxm = x_array[0] - x_array[1];
  dyp = *(array+2*n_columns+index_y) - *(array+1*n_columns+index_y);
  dym = *(array+0*n_columns+index_y) - *(array+1*n_columns+index_y);

  if ((dxp*dxm*(dxm-dxp)) == 0.) {
    sprintf(errmsg,"%s(L:%d) stop to avoid division by zero",__func__,__LINE__);
    return _FAILURE_;
  }

  *(array+1*n_columns+index_dy) = (dyp*dxm*dxm-dym*dxp*dxp)/(dxp*dxm*(dxm-dxp));

  *(array+0*n_columns+index_dy) = *(array+1*n_columns+index_dy)
    - (x_array[1] - x_array[0]) * 2.*(dyp*dxm-dym*dxp)/(dxp*dxm*(dxp-dxm));

  for (i=2; i<n_lines-1; i++) {

    dxp = x_array[i+1] - x_array[i];
    dxm = x_array[i-1] - x_array[i];
    dyp = *(array+(i+1)*n_columns+index_y) - *(array+i*n_columns+index_y);
    dym = *(array+(i-1)*n_columns+index_y) - *(array+i*n_columns+index_y);

    if ((dxp*dxm*(dxm-dxp)) == 0.) {
      sprintf(errmsg,"%s(L:%d) stop to avoid division by zero",__func__,__LINE__);
      return _FAILURE_;
    }

    *(array+i*n_columns+index_dy) = (dyp*dxm*dxm-dym*dxp*dxp)/(dxp*dxm*(dxm-dxp));

  }

  *(array+(n_lines-1)*n_columns+index_dy) = *(array+(n_lines-2)*n_columns+index_dy)
    + (x_array[n_lines-1] - x_array[n_lines-2]) * 2.*(dyp*dxm-dym*dxp)/(dxp*dxm*(dxp-dxm));

  return _SUCCESS_;

}

int array_derive2_order2_table_line_to_line(
				       double * x_array,
				       int n_lines,
				       double * array,
				       int n_columns,
				       int index_y,
				       int index_dy,
				       int index_ddy,
				       ErrorMsg errmsg) {

  int i;
  double dxp,dxm,dyp,dym;

  for (i=1; i<n_lines-1; i++) {

    dxp = x_array[i+1] - x_array[i];
    dxm = x_array[i-1] - x_array[i];
    dyp = *(array+(i+1)*n_columns+index_y) - *(array+i*n_columns+index_y);
    dym = *(array+(i-1)*n_columns+index_y) - *(array+i*n_columns+index_y);

    if ((dxp*dxm*(dxm-dxp)) == 0.) {
      sprintf(errmsg,"%s(L:%d) stop to avoid division by zero",__func__,__LINE__);
      return _FAILURE_;
    }

    *(array+i*n_columns+index_dy) = (dyp*dxm*dxm-dym*dxp*dxp)/(dxp*dxm*(dxm-dxp));
    *(array+i*n_columns+index_ddy) = 2.*(dyp*dxm-dym*dxp)/(dxp*dxm*(dxp-dxm));

  }

  *(array+0*n_columns+index_dy) = *(array+1*n_columns+index_dy)
    - (x_array[1] - x_array[0]) * *(array+1*n_columns+index_ddy);
  *(array+0*n_columns+index_ddy) = *(array+1*n_columns+index_ddy);

  *(array+(n_lines-1)*n_columns+index_dy) = *(array+(n_lines-2)*n_columns+index_dy)
    + (x_array[n_lines-1] - x_array[n_lines-2]) * *(array+(n_lines-2)*n_columns+index_ddy);
  *(array+(n_lines-1)*n_columns+index_ddy) = *(array+(n_lines-2)*n_columns+index_ddy);

  return _SUCCESS_;

}

int array_integrate_spline_table_line_to_line(
					      double * x_array,
					      int n_lines,
					      double * array,
					      int n_columns,
					      int index_y,
					      int index_ddy,
					      int index_inty,
					      ErrorMsg errmsg) {

  int i;

  double h;

  *(array+0*n_columns+index_inty)  = 0.;

  for (i=0; i < n_lines-1; i++) {

    h = (x_array[i+1]-x_array[i]);

    *(array+(i+1)*n_columns+index_inty) = *(array+i*n_columns+index_inty) +
      (array[i*n_columns+index_y]+array[(i+1)*n_columns+index_y])*h/2.+
      (array[i*n_columns+index_ddy]+array[(i+1)*n_columns+index_ddy])*h*h*h/24.;

  }

  return _SUCCESS_;
}


 /**
 * Not called.
 */
int array_derive_two(
		     double * array,
		     int n_columns,
		     int n_lines,
		     int index_x,   /** from 0 to (n_columns-1) */
		     int index_y,
		     int index_dydx,
		     int index_ddydxdx,
		     ErrorMsg errmsg) {

  int i;

  double dx1,dx2,dy1,dy2,weight1,weight2;

  if ((index_dydx == index_x) || (index_dydx == index_y)) {
    sprintf(errmsg,"%s(L:%d) : Output column %d must differ from input columns %d and %d",__func__,__LINE__,index_dydx,index_x,index_y);
    return _FAILURE_;
  }

  dx2=*(array+1*n_columns+index_x)-*(array+0*n_columns+index_x);
  dy2=*(array+1*n_columns+index_y)-*(array+0*n_columns+index_y);

  for (i=1; i<n_lines-1; i++) {

    dx1 = dx2;
    dy1 = dy2;
    dx2 = *(array+(i+1)*n_columns+index_x)-*(array+i*n_columns+index_x);
    dy2 = *(array+(i+1)*n_columns+index_y)-*(array+i*n_columns+index_y);
    weight1 = dx2*dx2;
    weight2 = dx1*dx1;

    if ((dx1 == 0.) && (dx2 == 0.)) {
      sprintf(errmsg,"%s(L:%d) stop to avoid division by zero",__func__,__LINE__);
      return _FAILURE_;
    }

    *(array+i*n_columns+index_dydx) = (weight1*dy1+weight2*dy2) / (weight1*dx1+weight2*dx2);
    *(array+i*n_columns+index_ddydxdx) = (dx2*dy1-dx1*dy2) / (weight1*dx1+weight2*dx2);

    if (i == 1) {
      *(array+(i-1)*n_columns+index_dydx) = 2.*dy1/dx1 - *(array+i*n_columns+index_dydx);
      *(array+(i-1)*n_columns+index_ddydxdx) = *(array+i*n_columns+index_ddydxdx);
    }

    if (i == n_lines-2) {
      *(array+(i+1)*n_columns+index_dydx) = 2.*dy2/dx2 - *(array+i*n_columns+index_dydx);
      *(array+(i+1)*n_columns+index_dydx) = *(array+i*n_columns+index_ddydxdx);
    }
  }

  return _SUCCESS_;
}

int array_spline(
		  double * array,
		  int n_columns,
		  int n_lines,
		  int index_x,   /** from 0 to (n_columns-1) */
		  int index_y,
		  int index_ddydx2,
		  short spline_mode,
		  ErrorMsg errmsg) {

  int i,k;
  double p,qn,sig,un;
  double * u;
  double dy_first;
  double dy_last;

  if (n_lines < 3) {
    sprintf(errmsg,"%s(L:%d) n_lines=%d, while routine needs n_lines >= 3",__func__,__LINE__,n_lines);
    return _FAILURE_;
  }

  u = malloc((n_lines-1) * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }

  if (spline_mode == _SPLINE_NATURAL_) {
    *(array+0*n_columns+index_ddydx2) = u[0] = 0.0;
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {
      dy_first =
	((*(array+2*n_columns+index_x)-*(array+0*n_columns+index_x))*
	 (*(array+2*n_columns+index_x)-*(array+0*n_columns+index_x))*
	 (*(array+1*n_columns+index_y)-*(array+0*n_columns+index_y))-
	 (*(array+1*n_columns+index_x)-*(array+0*n_columns+index_x))*
	 (*(array+1*n_columns+index_x)-*(array+0*n_columns+index_x))*
	 (*(array+2*n_columns+index_y)-*(array+0*n_columns+index_y)))/
	((*(array+2*n_columns+index_x)-*(array+0*n_columns+index_x))*
	 (*(array+1*n_columns+index_x)-*(array+0*n_columns+index_x))*
	 (*(array+2*n_columns+index_x)-*(array+1*n_columns+index_x)));

      *(array+0*n_columns+index_ddydx2) = -0.5;

      u[0] =
	(3./(*(array+1*n_columns+index_x) -  *(array+0*n_columns+index_x)))*
	((*(array+1*n_columns+index_y) -  *(array+0*n_columns+index_y))/
	 (*(array+1*n_columns+index_x) -  *(array+0*n_columns+index_x))
	 -dy_first);
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  for (i=1; i < n_lines-1; i++) {

      sig = (*(array+i*n_columns+index_x) - *(array+(i-1)*n_columns+index_x))
	/ (*(array+(i+1)*n_columns+index_x) - *(array+(i-1)*n_columns+index_x));

      p = sig * *(array+(i-1)*n_columns+index_ddydx2) + 2.0;
      *(array+i*n_columns+index_ddydx2) = (sig-1.0)/p;
      u[i] = (*(array+(i+1)*n_columns+index_y) - *(array+i*n_columns+index_y))
	/ (*(array+(i+1)*n_columns+index_x) - *(array+i*n_columns+index_x))
	- (*(array+i*n_columns+index_y) - *(array+(i-1)*n_columns+index_y))
	/ (*(array+i*n_columns+index_x) - *(array+(i-1)*n_columns+index_x));
      u[i]= (6.0 * u[i] /
	     (*(array+(i+1)*n_columns+index_x) - *(array+(i-1)*n_columns+index_x))
	     - sig * u[i-1]) / p;

    }

  if (spline_mode == _SPLINE_NATURAL_) {
    qn=0.;
    un=0.;
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {
      dy_last =
	((*(array+(n_lines-3)*n_columns+index_x)-*(array+(n_lines-1)*n_columns+index_x))*
	 (*(array+(n_lines-3)*n_columns+index_x)-*(array+(n_lines-1)*n_columns+index_x))*
	 (*(array+(n_lines-2)*n_columns+index_y)-*(array+(n_lines-1)*n_columns+index_y))-
	 (*(array+(n_lines-2)*n_columns+index_x)-*(array+(n_lines-1)*n_columns+index_x))*
	 (*(array+(n_lines-2)*n_columns+index_x)-*(array+(n_lines-1)*n_columns+index_x))*
	 (*(array+(n_lines-3)*n_columns+index_y)-*(array+(n_lines-1)*n_columns+index_y)))/
	((*(array+(n_lines-3)*n_columns+index_x)-*(array+(n_lines-1)*n_columns+index_x))*
	 (*(array+(n_lines-2)*n_columns+index_x)-*(array+(n_lines-1)*n_columns+index_x))*
	 (*(array+(n_lines-3)*n_columns+index_x)-*(array+(n_lines-2)*n_columns+index_x)));

      qn=0.5;
      un =
	(3./(*(array+(n_lines-1)*n_columns+index_x) -  *(array+(n_lines-2)*n_columns+index_x)))*
	(dy_last-(*(array+(n_lines-1)*n_columns+index_y) -  *(array+(n_lines-2)*n_columns+index_y))/
	 (*(array+(n_lines-1)*n_columns+index_x) -  *(array+(n_lines-2)*n_columns+index_x)));
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  *(array+(n_lines-1)*n_columns+index_ddydx2) =
    (un-qn*u[n_lines-2])/(qn* *(array+(n_lines-2)*n_columns+index_ddydx2)+1.0);

  for (k=n_lines-2; k>=0; k--)
    *(array+k*n_columns+index_ddydx2) = *(array+k*n_columns+index_ddydx2) *
      *(array+(k+1)*n_columns+index_ddydx2) + u[k];

  free(u);

  return _SUCCESS_;
}

int array_spline_table_line_to_line(
				    double * x, /* vector of size x_size */
				    int n_lines,
				    double * array,
				    int n_columns,
				    int index_y,
				    int index_ddydx2,
				    short spline_mode,
				    ErrorMsg errmsg) {

  int i,k;
  double p,qn,sig,un;
  double * u;
  double dy_first;
  double dy_last;

  u = malloc((n_lines-1) * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }

  if (spline_mode == _SPLINE_NATURAL_) {
    *(array+0*n_columns+index_ddydx2) = u[0] = 0.0;
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {
      dy_first =
	((x[2]-x[0])*(x[2]-x[0])*
	 (*(array+1*n_columns+index_y)-*(array+0*n_columns+index_y))-
	 (x[1]-x[0])*(x[1]-x[0])*
	 (*(array+2*n_columns+index_y)-*(array+0*n_columns+index_y)))/
	((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]));
      *(array+0*n_columns+index_ddydx2) = -0.5;
      u[0] =
	(3./(x[1] -  x[0]))*
	((*(array+1*n_columns+index_y) -  *(array+0*n_columns+index_y))/
	 (x[1] - x[0])-dy_first);
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  for (i=1; i < n_lines-1; i++) {

      sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1]);

      p = sig * *(array+(i-1)*n_columns+index_ddydx2) + 2.0;
      *(array+i*n_columns+index_ddydx2) = (sig-1.0)/p;
      u[i] = (*(array+(i+1)*n_columns+index_y) - *(array+i*n_columns+index_y))
	/ (x[i+1] - x[i])
	- (*(array+i*n_columns+index_y) - *(array+(i-1)*n_columns+index_y))
	/ (x[i] - x[i-1]);
      u[i]= (6.0 * u[i] /
	     (x[i+1] - x[i-1])
	     - sig * u[i-1]) / p;

  }

  if (spline_mode == _SPLINE_NATURAL_) {
    qn=0.;
    un=0.;
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {
      dy_last =
	((x[n_lines-3]-x[n_lines-1])*(x[n_lines-3]-x[n_lines-1])*
	 (*(array+(n_lines-2)*n_columns+index_y)-*(array+(n_lines-1)*n_columns+index_y))-
	 (x[n_lines-2]-x[n_lines-1])*(x[n_lines-2]-x[n_lines-1])*
	 (*(array+(n_lines-3)*n_columns+index_y)-*(array+(n_lines-1)*n_columns+index_y)))/
	((x[n_lines-3]-x[n_lines-1])*(x[n_lines-2]-x[n_lines-1])*(x[n_lines-3]-x[n_lines-2]));
      qn=0.5;
      un =
	(3./(x[n_lines-1] - x[n_lines-2]))*
	(dy_last-(*(array+(n_lines-1)*n_columns+index_y) -  *(array+(n_lines-2)*n_columns+index_y))/
	 (x[n_lines-1] - x[n_lines-2]));
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  *(array+(n_lines-1)*n_columns+index_ddydx2) =
    (un-qn*u[n_lines-2])/(qn* *(array+(n_lines-2)*n_columns+index_ddydx2)+1.0);

  for (k=n_lines-2; k>=0; k--)
    *(array+k*n_columns+index_ddydx2) = *(array+k*n_columns+index_ddydx2) *
      *(array+(k+1)*n_columns+index_ddydx2) + u[k];

  free(u);

  return _SUCCESS_;
 }

int array_spline_table_lines(
			     double * x, /* vector of size x_size */
			     int x_size,
			     double * y_array, /* array of size x_size*y_size with elements
						  y_array[index_x*y_size+index_y] */
			     int y_size,
			     double * ddy_array, /* array of size x_size*y_size */
			     short spline_mode,
			     ErrorMsg errmsg
			     ) {

  double * p;
  double * qn;
  double * un;
  double * u;
  double sig;
  int index_x;
  int index_y;
  double dy_first;
  double dy_last;

  u = malloc((x_size-1) * y_size * sizeof(double));
  p = malloc(y_size * sizeof(double));
  qn = malloc(y_size * sizeof(double));
  un = malloc(y_size * sizeof(double));

  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }
  if (p == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate p",__func__,__LINE__);
    return _FAILURE_;
  }
  if (qn == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate qn",__func__,__LINE__);
    return _FAILURE_;
  }
  if (un == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate un",__func__,__LINE__);
    return _FAILURE_;
  }

  if (x_size==2) spline_mode = _SPLINE_NATURAL_; // in the case of only 2 x-values, only the natural spline method is appropriate, for _SPLINE_EST_DERIV_ at least 3 x-values are needed.


  index_x=0;

  if (spline_mode == _SPLINE_NATURAL_) {
    for (index_y=0; index_y < y_size; index_y++) {
      ddy_array[index_x*y_size+index_y] = u[index_x*y_size+index_y] = 0.0;
    }
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      for (index_y=0; index_y < y_size; index_y++) {

	dy_first =
	  ((x[2]-x[0])*(x[2]-x[0])*
	   (y_array[1*y_size+index_y]-y_array[0*y_size+index_y])-
	   (x[1]-x[0])*(x[1]-x[0])*
	   (y_array[2*y_size+index_y]-y_array[0*y_size+index_y]))/
	  ((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]));

	ddy_array[index_x*y_size+index_y] = -0.5;

	u[index_x*y_size+index_y] =
	  (3./(x[1] -  x[0]))*
	  ((y_array[1*y_size+index_y]-y_array[0*y_size+index_y])/
	   (x[1] - x[0])-dy_first);

      }
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }


  for (index_x=1; index_x < x_size-1; index_x++) {

    sig = (x[index_x] - x[index_x-1])/(x[index_x+1] - x[index_x-1]);

    for (index_y=0; index_y < y_size; index_y++) {

      p[index_y] = sig * ddy_array[(index_x-1)*y_size+index_y] + 2.0;

      ddy_array[index_x*y_size+index_y] = (sig-1.0)/p[index_y];

      u[index_x*y_size+index_y] =
	(y_array[(index_x+1)*y_size+index_y] - y_array[index_x*y_size+index_y])
	/ (x[index_x+1] - x[index_x])
	- (y_array[index_x*y_size+index_y] - y_array[(index_x-1)*y_size+index_y])
	/ (x[index_x] - x[index_x-1]);

      u[index_x*y_size+index_y] = (6.0 * u[index_x*y_size+index_y] /
				   (x[index_x+1] - x[index_x-1])
				   - sig * u[(index_x-1)*y_size+index_y]) / p[index_y];
    }

  }

  if (spline_mode == _SPLINE_NATURAL_) {

    for (index_y=0; index_y < y_size; index_y++) {
      qn[index_y]=un[index_y]=0.0;
    }

  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      for (index_y=0; index_y < y_size; index_y++) {

	dy_last =
	  ((x[x_size-3]-x[x_size-1])*(x[x_size-3]-x[x_size-1])*
	   (y_array[(x_size-2)*y_size+index_y]-y_array[(x_size-1)*y_size+index_y])-
	   (x[x_size-2]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*
	   (y_array[(x_size-3)*y_size+index_y]-y_array[(x_size-1)*y_size+index_y]))/
	  ((x[x_size-3]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*(x[x_size-3]-x[x_size-2]));

	qn[index_y]=0.5;

	un[index_y]=
	  (3./(x[x_size-1] - x[x_size-2]))*
	  (dy_last-(y_array[(x_size-1)*y_size+index_y] - y_array[(x_size-2)*y_size+index_y])/
	   (x[x_size-1] - x[x_size-2]));

      }
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  index_x=x_size-1;

  for (index_y=0; index_y < y_size; index_y++) {
    ddy_array[index_x*y_size+index_y] =
      (un[index_y] - qn[index_y] * u[(index_x-1)*y_size+index_y]) /
      (qn[index_y] * ddy_array[(index_x-1)*y_size+index_y] + 1.0);
  }

  for (index_x=x_size-2; index_x >= 0; index_x--) {
    for (index_y=0; index_y < y_size; index_y++) {

      ddy_array[index_x*y_size+index_y] = ddy_array[index_x*y_size+index_y] *
	ddy_array[(index_x+1)*y_size+index_y] + u[index_x*y_size+index_y];

    }
  }

  free(qn);
  free(un);
  free(p);
  free(u);

  return _SUCCESS_;
 }

int array_logspline_table_lines(
			     double * x, /* vector of size x_size */
			     int x_size,
			     double * y_array, /* array of size x_size*y_size with elements
						  y_array[index_x*y_size+index_y] */
			     int y_size,
			     double * ddlny_array, /* array of size x_size*y_size */
			     short spline_mode,
			     ErrorMsg errmsg
			     ) {

  double * p;
  double * qn;
  double * un;
  double * u;
  double sig;
  int index_x;
  int index_y;
  double dy_first;
  double dy_last;

  u = malloc((x_size-1) * y_size * sizeof(double));
  p = malloc(y_size * sizeof(double));
  qn = malloc(y_size * sizeof(double));
  un = malloc(y_size * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }
  if (p == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate p",__func__,__LINE__);
    return _FAILURE_;
  }
  if (qn == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate qn",__func__,__LINE__);
    return _FAILURE_;
  }
  if (un == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate un",__func__,__LINE__);
    return _FAILURE_;
  }

  if (x_size==2) spline_mode = _SPLINE_NATURAL_; // in the case of only 2 x-values, only the natural spline method is appropriate, for _SPLINE_EST_DERIV_ at least 3 x-values are needed.


  index_x=0;

  if (spline_mode == _SPLINE_NATURAL_) {
    for (index_y=0; index_y < y_size; index_y++) {
      ddlny_array[index_x*y_size+index_y] = u[index_x*y_size+index_y] = 0.0;
    }
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      for (index_y=0; index_y < y_size; index_y++) {

	dy_first =
	  ((log(x[2])-log(x[0]))*(log(x[2])-log(x[0]))*
	   (log(y_array[1*y_size+index_y])-log(y_array[0*y_size+index_y]))-
	   (log(x[1])-log(x[0]))*(log(x[1])-log(x[0]))*
	   (log(y_array[2*y_size+index_y])-log(y_array[0*y_size+index_y])))/
	  ((log(x[2])-log(x[0]))*(log(x[1])-log(x[0]))*(log(x[2])-log(x[1])));

	ddlny_array[index_x*y_size+index_y] = -0.5;

	u[index_x*y_size+index_y] =
	  (3./(log(x[1]) - log(x[0])))*
	  ((log(y_array[1*y_size+index_y])-log(y_array[0*y_size+index_y]))/
	   (log(x[1]) - log(x[0]))-dy_first);

      }
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }


  for (index_x=1; index_x < x_size-1; index_x++) {

    sig = (log(x[index_x]) - log(x[index_x-1]))/(log(x[index_x+1]) - log(x[index_x-1]));

    for (index_y=0; index_y < y_size; index_y++) {

      p[index_y] = sig * ddlny_array[(index_x-1)*y_size+index_y] + 2.0;

      ddlny_array[index_x*y_size+index_y] = (sig-1.0)/p[index_y];

      u[index_x*y_size+index_y] =
	(log(y_array[(index_x+1)*y_size+index_y]) - log(y_array[index_x*y_size+index_y]))
	/ (log(x[index_x+1]) - log(x[index_x]))
	- (log(y_array[index_x*y_size+index_y]) - log(y_array[(index_x-1)*y_size+index_y]))
	/ (log(x[index_x]) - log(x[index_x-1]));

      u[index_x*y_size+index_y] = (6.0 * u[index_x*y_size+index_y] /
				   (log(x[index_x+1]) - log(x[index_x-1]))
				   - sig * u[(index_x-1)*y_size+index_y]) / p[index_y];
    }

  }

  if (spline_mode == _SPLINE_NATURAL_) {

    for (index_y=0; index_y < y_size; index_y++) {
      qn[index_y]=un[index_y]=0.0;
    }

  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      for (index_y=0; index_y < y_size; index_y++) {

	dy_last =
	  ((log(x[x_size-3])-log(x[x_size-1]))*(log(x[x_size-3])-log(x[x_size-1]))*
	   (log(y_array[(x_size-2)*y_size+index_y])-log(y_array[(x_size-1)*y_size+index_y]))-
	   (log(x[x_size-2])-log(x[x_size-1]))*(log(x[x_size-2])-log(x[x_size-1]))*
	   (log(y_array[(x_size-3)*y_size+index_y])-log(y_array[(x_size-1)*y_size+index_y])))/
	  ((log(x[x_size-3])-log(x[x_size-1]))*(log(x[x_size-2])-log(x[x_size-1]))*(log(x[x_size-3])-log(x[x_size-2])));

	qn[index_y]=0.5;

	un[index_y]=
	  (3./(log(x[x_size-1]) - log(x[x_size-2])))*
	  (dy_last-(log(y_array[(x_size-1)*y_size+index_y]) - log(y_array[(x_size-2)*y_size+index_y]))/
	   (log(x[x_size-1]) - log(x[x_size-2])));

      }
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  index_x=x_size-1;


  for (index_y=0; index_y < y_size; index_y++) {
    ddlny_array[index_x*y_size+index_y] =
      (un[index_y] - qn[index_y] * u[(index_x-1)*y_size+index_y]) /
      (qn[index_y] * ddlny_array[(index_x-1)*y_size+index_y] + 1.0);
  }

  for (index_x=x_size-2; index_x >= 0; index_x--) {
    for (index_y=0; index_y < y_size; index_y++) {

      ddlny_array[index_x*y_size+index_y] = ddlny_array[index_x*y_size+index_y] *
	ddlny_array[(index_x+1)*y_size+index_y] + u[index_x*y_size+index_y];

    }
  }

  free(qn);
  free(un);
  free(p);
  free(u);

  return _SUCCESS_;
 }

int array_spline_table_columns(
		       double * x, /* vector of size x_size */
		       int x_size,
		       double * y_array, /* array of size x_size*y_size with elements
					  y_array[index_y*x_size+index_x] */
		       int y_size,
		       double * ddy_array, /* array of size x_size*y_size */
		       short spline_mode,
		       ErrorMsg errmsg
		       ) {

  double * p;
  double * qn;
  double * un;
  double * u;
  double sig;
  int index_x;
  int index_y;
  double dy_first;
  double dy_last;

  u = malloc((x_size-1) * y_size * sizeof(double));
  p = malloc(y_size * sizeof(double));
  qn = malloc(y_size * sizeof(double));
  un = malloc(y_size * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }
  if (p == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate p",__func__,__LINE__);
    return _FAILURE_;
  }
  if (qn == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate qn",__func__,__LINE__);
    return _FAILURE_;
  }
  if (un == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate un",__func__,__LINE__);
    return _FAILURE_;
  }

  if (x_size==2) spline_mode = _SPLINE_NATURAL_; // in the case of only 2 x-values, only the natural spline method is appropriate, for _SPLINE_EST_DERIV_ at least 3 x-values are needed.

  index_x=0;

  if (spline_mode == _SPLINE_NATURAL_) {
    for (index_y=0; index_y < y_size; index_y++) {
      ddy_array[index_y*x_size+index_x] = 0.0;
      u[index_x*y_size+index_y] = 0.0;
    }
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      class_test(x[2]-x[0]==0.,
		 errmsg,
		 "x[2]=%g, x[0]=%g, stop to avoid seg fault",x[2],x[0]);
      class_test(x[1]-x[0]==0.,
		 errmsg,
		 "x[1]=%g, x[0]=%g, stop to avoid seg fault",x[1],x[0]);
      class_test(x[2]-x[1]==0.,
		 errmsg,
		 "x[2]=%g, x[1]=%g, stop to avoid seg fault",x[2],x[1]);

      for (index_y=0; index_y < y_size; index_y++) {

	dy_first =
	  ((x[2]-x[0])*(x[2]-x[0])*
	   (y_array[index_y*x_size+1]-y_array[index_y*x_size+0])-
	   (x[1]-x[0])*(x[1]-x[0])*
	   (y_array[index_y*x_size+2]-y_array[index_y*x_size+0]))/
	  ((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]));

	ddy_array[index_y*x_size+index_x] = -0.5;

	u[index_x*y_size+index_y] =
	  (3./(x[1] -  x[0]))*
	  ((y_array[index_y*x_size+1]-y_array[index_y*x_size+0])/
	   (x[1] - x[0])-dy_first);

      }
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  for (index_x=1; index_x < x_size-1; index_x++) {

    sig = (x[index_x] - x[index_x-1])/(x[index_x+1] - x[index_x-1]);

    for (index_y=0; index_y < y_size; index_y++) {

      p[index_y] = sig * ddy_array[index_y*x_size+(index_x-1)] + 2.0;

      ddy_array[index_y*x_size+index_x] = (sig-1.0)/p[index_y];

      u[index_x*y_size+index_y] =
	(y_array[index_y*x_size+(index_x+1)] - y_array[index_y*x_size+index_x])
	/ (x[index_x+1] - x[index_x])
	- (y_array[index_y*x_size+index_x] - y_array[index_y*x_size+(index_x-1)])
	/ (x[index_x] - x[index_x-1]);

      u[index_x*y_size+index_y] = (6.0 * u[index_x*y_size+index_y] /
				   (x[index_x+1] - x[index_x-1])
				   - sig * u[(index_x-1)*y_size+index_y]) / p[index_y];
    }

  }

  if (spline_mode == _SPLINE_NATURAL_) {

    for (index_y=0; index_y < y_size; index_y++) {
      qn[index_y]=un[index_y]=0.0;
    }

  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      for (index_y=0; index_y < y_size; index_y++) {

	dy_last =
	  ((x[x_size-3]-x[x_size-1])*(x[x_size-3]-x[x_size-1])*
	   (y_array[index_y*x_size+(x_size-2)]-y_array[index_y*x_size+(x_size-1)])-
	   (x[x_size-2]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*
	   (y_array[index_y*x_size+(x_size-3)]-y_array[index_y*x_size+(x_size-1)]))/
	  ((x[x_size-3]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*(x[x_size-3]-x[x_size-2]));

	qn[index_y]=0.5;

	un[index_y]=
	  (3./(x[x_size-1] - x[x_size-2]))*
	  (dy_last-(y_array[index_y*x_size+(x_size-1)] - y_array[index_y*x_size+(x_size-2)])/
	   (x[x_size-1] - x[x_size-2]));

      }
    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  index_x=x_size-1;

  for (index_y=0; index_y < y_size; index_y++) {
    ddy_array[index_y*x_size+index_x] =
      (un[index_y] - qn[index_y] * u[(index_x-1)*y_size+index_y]) /
      (qn[index_y] * ddy_array[index_y*x_size+(index_x-1)] + 1.0);
  }

  for (index_x=x_size-2; index_x >= 0; index_x--) {
    for (index_y=0; index_y < y_size; index_y++) {

      ddy_array[index_y*x_size+index_x] = ddy_array[index_y*x_size+index_x] *
	ddy_array[index_y*x_size+(index_x+1)] + u[index_x*y_size+index_y];

    }
  }

  free(qn);
  free(p);
  free(u);
  free(un);

  return _SUCCESS_;
 }

int array_spline_table_columns2(
		       double * x, /* vector of size x_size */
		       int x_size,
		       double * y_array, /* array of size x_size*y_size with elements
					  y_array[index_y*x_size+index_x] */
		       int y_size,
		       double * ddy_array, /* array of size x_size*y_size */
		       short spline_mode,
		       ErrorMsg errmsg
		       ) {

  double * p;
  double * qn;
  double * un;
  double * u;
  double sig;
  int index_x;
  int index_y;
  double dy_first;
  double dy_last;

  u = malloc((x_size-1) * y_size * sizeof(double));
  p = malloc(y_size * sizeof(double));
  qn = malloc(y_size * sizeof(double));
  un = malloc(y_size * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }
  if (p == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate p",__func__,__LINE__);
    return _FAILURE_;
  }
  if (qn == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate qn",__func__,__LINE__);
    return _FAILURE_;
  }
  if (un == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate un",__func__,__LINE__);
    return _FAILURE_;
  }

  if (x_size==2) spline_mode = _SPLINE_NATURAL_; // in the case of only 2 x-values, only the natural spline method is appropriate, for _SPLINE_EST_DERIV_ at least 3 x-values are needed.

#pragma omp parallel                                                \
  shared(x,x_size,y_array,y_size,ddy_array,spline_mode,p,qn,un,u)   \
  private(index_y,index_x,sig,dy_first,dy_last)
  {

#pragma omp for schedule (dynamic)

    for (index_y=0; index_y < y_size; index_y++) {

      if (spline_mode == _SPLINE_NATURAL_) {
        ddy_array[index_y*x_size+0] = 0.0;
        u[0*y_size+index_y] = 0.0;
      }
      else {
        dy_first =
          ((x[2]-x[0])*(x[2]-x[0])*
           (y_array[index_y*x_size+1]-y_array[index_y*x_size+0])-
           (x[1]-x[0])*(x[1]-x[0])*
           (y_array[index_y*x_size+2]-y_array[index_y*x_size+0]))/
          ((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]));

        ddy_array[index_y*x_size+0] = -0.5;

        u[0*y_size+index_y] =
          (3./(x[1] -  x[0]))*
          ((y_array[index_y*x_size+1]-y_array[index_y*x_size+0])/
           (x[1] - x[0])-dy_first);

      }

      for (index_x=1; index_x < x_size-1; index_x++) {

        sig = (x[index_x] - x[index_x-1])/(x[index_x+1] - x[index_x-1]);

        p[index_y] = sig * ddy_array[index_y*x_size+(index_x-1)] + 2.0;

        ddy_array[index_y*x_size+index_x] = (sig-1.0)/p[index_y];

        u[index_x*y_size+index_y] =
          (y_array[index_y*x_size+(index_x+1)] - y_array[index_y*x_size+index_x])
          / (x[index_x+1] - x[index_x])
          - (y_array[index_y*x_size+index_x] - y_array[index_y*x_size+(index_x-1)])
          / (x[index_x] - x[index_x-1]);

        u[index_x*y_size+index_y] = (6.0 * u[index_x*y_size+index_y] /
                                     (x[index_x+1] - x[index_x-1])
                                     - sig * u[(index_x-1)*y_size+index_y]) / p[index_y];

      }

      if (spline_mode == _SPLINE_NATURAL_) {

        qn[index_y]=un[index_y]=0.0;

      }
      else {

        dy_last =
          ((x[x_size-3]-x[x_size-1])*(x[x_size-3]-x[x_size-1])*
           (y_array[index_y*x_size+(x_size-2)]-y_array[index_y*x_size+(x_size-1)])-
           (x[x_size-2]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*
           (y_array[index_y*x_size+(x_size-3)]-y_array[index_y*x_size+(x_size-1)]))/
          ((x[x_size-3]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*(x[x_size-3]-x[x_size-2]));

        qn[index_y]=0.5;

        un[index_y]=
          (3./(x[x_size-1] - x[x_size-2]))*
          (dy_last-(y_array[index_y*x_size+(x_size-1)] - y_array[index_y*x_size+(x_size-2)])/
           (x[x_size-1] - x[x_size-2]));

      }

      index_x=x_size-1;

      ddy_array[index_y*x_size+index_x] =
        (un[index_y] - qn[index_y] * u[(index_x-1)*y_size+index_y]) /
        (qn[index_y] * ddy_array[index_y*x_size+(index_x-1)] + 1.0);

      for (index_x=x_size-2; index_x >= 0; index_x--) {

        ddy_array[index_y*x_size+index_x] = ddy_array[index_y*x_size+index_x] *
          ddy_array[index_y*x_size+(index_x+1)] + u[index_x*y_size+index_y];

      }
    }
  }
  free(qn);
  free(p);
  free(u);
  free(un);

  return _SUCCESS_;
 }

int array_spline_table_one_column(
		       double * x, /* vector of size x_size */
		       int x_size,
		       double * y_array, /* array of size x_size*y_size with elements
					  y_array[index_y*x_size+index_x] */
		       int y_size,
		       int index_y,
		       double * ddy_array, /* array of size x_size*y_size */
		       short spline_mode,
		       ErrorMsg errmsg
		       ) {

  double p;
  double qn;
  double un;
  double * u;
  double sig;
  int index_x;
  double dy_first;
  double dy_last;

  u = malloc((x_size-1) * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }

  if (x_size==2) spline_mode = _SPLINE_NATURAL_; // in the case of only 2 x-values, only the natural spline method is appropriate, for _SPLINE_EST_DERIV_ at least 3 x-values are needed.

  /************************************************/

  index_x=0;

  if (spline_mode == _SPLINE_NATURAL_) {
    ddy_array[index_y*x_size+index_x] = 0.0;
    u[index_x] = 0.0;
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      dy_first =
	((x[2]-x[0])*(x[2]-x[0])*
	 (y_array[index_y*x_size+1]-y_array[index_y*x_size+0])-
	 (x[1]-x[0])*(x[1]-x[0])*
	 (y_array[index_y*x_size+2]-y_array[index_y*x_size+0]))/
	((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]));

      ddy_array[index_y*x_size+index_x] = -0.5;

      u[index_x] =
	(3./(x[1] -  x[0]))*
	((y_array[index_y*x_size+1]-y_array[index_y*x_size+0])/
	 (x[1] - x[0])-dy_first);

    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  /************************************************/

  for (index_x=1; index_x < x_size-1; index_x++) {

    sig = (x[index_x] - x[index_x-1])/(x[index_x+1] - x[index_x-1]);

    p = sig * ddy_array[index_y*x_size+(index_x-1)] + 2.0;

    ddy_array[index_y*x_size+index_x] = (sig-1.0)/p;

    u[index_x] =
      (y_array[index_y*x_size+(index_x+1)] - y_array[index_y*x_size+index_x])
      / (x[index_x+1] - x[index_x])
      - (y_array[index_y*x_size+index_x] - y_array[index_y*x_size+(index_x-1)])
      / (x[index_x] - x[index_x-1]);

    u[index_x] = (6.0 * u[index_x] /
		  (x[index_x+1] - x[index_x-1])
		  - sig * u[index_x-1]) / p;

  }

  /************************************************/

  if (spline_mode == _SPLINE_NATURAL_) {

      qn=un=0.0;

  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      dy_last =
	((x[x_size-3]-x[x_size-1])*(x[x_size-3]-x[x_size-1])*
	 (y_array[index_y*x_size+(x_size-2)]-y_array[index_y*x_size+(x_size-1)])-
	 (x[x_size-2]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*
	 (y_array[index_y*x_size+(x_size-3)]-y_array[index_y*x_size+(x_size-1)]))/
	((x[x_size-3]-x[x_size-1])*(x[x_size-2]-x[x_size-1])*(x[x_size-3]-x[x_size-2]));

      qn=0.5;

      un=
	(3./(x[x_size-1] - x[x_size-2]))*
	(dy_last-(y_array[index_y*x_size+(x_size-1)] - y_array[index_y*x_size+(x_size-2)])/
	 (x[x_size-1] - x[x_size-2]));

    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  /************************************************/

  index_x=x_size-1;

  ddy_array[index_y*x_size+index_x] =
    (un - qn * u[index_x-1]) /
    (qn * ddy_array[index_y*x_size+(index_x-1)] + 1.0);

  for (index_x=x_size-2; index_x >= 0; index_x--) {

    ddy_array[index_y*x_size+index_x] = ddy_array[index_y*x_size+index_x] *
      ddy_array[index_y*x_size+(index_x+1)] + u[index_x];

  }

  free(u);

  return _SUCCESS_;
}

int array_logspline_table_one_column(
		       double * x, /* vector of size x_size */
		       int x_size,
		       int x_stop,
		       double * y_array, /* array of size x_size*y_size with elements
					  y_array[index_y*x_size+index_x] */
		       int y_size,
		       int index_y,
		       double * ddlogy_array, /* array of size x_size*y_size */
		       short spline_mode,
		       ErrorMsg errmsg
		       ) {

  double p;
  double qn;
  double un;
  double * u;
  double sig;
  int index_x;
  double dy_first;
  double dy_last;

  u = malloc((x_stop-1) * sizeof(double));
  if (u == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate u",__func__,__LINE__);
    return _FAILURE_;
  }

  if (x_size==2) spline_mode = _SPLINE_NATURAL_; // in the case of only 2 x-values, only the natural spline method is appropriate, for _SPLINE_EST_DERIV_ at least 3 x-values are needed.

  /************************************************/

  index_x=0;

  if (spline_mode == _SPLINE_NATURAL_) {
    ddlogy_array[index_y*x_size+index_x] = 0.0;
    u[index_x] = 0.0;
  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      dy_first =
	((log(x[2])-log(x[0]))*(log(x[2])-log(x[0]))*
	 (log(y_array[index_y*x_size+1])-log(y_array[index_y*x_size+0]))-
	 (log(x[1])-log(x[0]))*(log(x[1])-log(x[0]))*
	 (log(y_array[index_y*x_size+2])-log(y_array[index_y*x_size+0])))/
	((log(x[2])-log(x[0]))*(log(x[1])-log(x[0]))*(log(x[2])-log(x[1])));

      ddlogy_array[index_y*x_size+index_x] = -0.5;

      u[index_x] =
	(3./(log(x[1]) -  log(x[0])))*
	((log(y_array[index_y*x_size+1])-log(y_array[index_y*x_size+0]))/
	 (log(x[1]) - log(x[0]))-dy_first);

    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  /************************************************/

  for (index_x=1; index_x < x_stop-1; index_x++) {

    sig = (log(x[index_x]) - log(x[index_x-1]))/(log(x[index_x+1]) - log(x[index_x-1]));

    p = sig * ddlogy_array[index_y*x_size+(index_x-1)] + 2.0;

    ddlogy_array[index_y*x_size+index_x] = (sig-1.0)/p;

    u[index_x] =
      (log(y_array[index_y*x_size+(index_x+1)]) - log(y_array[index_y*x_size+index_x]))
      / (log(x[index_x+1]) - log(x[index_x]))
      - (log(y_array[index_y*x_size+index_x]) - log(y_array[index_y*x_size+(index_x-1)]))
      / (log(x[index_x]) - log(x[index_x-1]));

    u[index_x] = (6.0 * u[index_x] /
		  (log(x[index_x+1]) - log(x[index_x-1]))
		  - sig * u[index_x-1]) / p;

  }

  /************************************************/

  if (spline_mode == _SPLINE_NATURAL_) {

      qn=un=0.0;

  }
  else {
    if (spline_mode == _SPLINE_EST_DERIV_) {

      dy_last =
	((log(x[x_stop-3])-log(x[x_stop-1]))*(log(x[x_stop-3])-log(x[x_stop-1]))*
	 (log(y_array[index_y*x_size+(x_stop-2)])-log(y_array[index_y*x_size+(x_stop-1)]))-
	 (log(x[x_stop-2])-log(x[x_stop-1]))*(log(x[x_stop-2])-log(x[x_stop-1]))*
	 (log(y_array[index_y*x_size+(x_stop-3)])-log(y_array[index_y*x_size+(x_stop-1)])))/
	((log(x[x_stop-3])-log(x[x_stop-1]))*(log(x[x_stop-2])-log(x[x_stop-1]))*
	 (log(x[x_stop-3])-log(x[x_stop-2])));

      qn=0.5;

      un=
	(3./(log(x[x_stop-1]) - log(x[x_stop-2])))*
	(dy_last-(log(y_array[index_y*x_size+(x_stop-1)]) - log(y_array[index_y*x_size+(x_stop-2)]))/
	 (log(x[x_stop-1]) - log(x[x_stop-2])));

    }
    else {
      sprintf(errmsg,"%s(L:%d) Spline mode not identified: %d",__func__,__LINE__,spline_mode);
      return _FAILURE_;
    }
  }

  /************************************************/

  index_x=x_stop-1;

  ddlogy_array[index_y*x_size+index_x] =
    (un - qn * u[index_x-1]) /
    (qn * ddlogy_array[index_y*x_size+(index_x-1)] + 1.0);

  for (index_x=x_stop-2; index_x >= 0; index_x--) {

    ddlogy_array[index_y*x_size+index_x] = ddlogy_array[index_y*x_size+index_x] *
      ddlogy_array[index_y*x_size+(index_x+1)] + u[index_x];

  }

  free(u);

  return _SUCCESS_;
}

int array_integrate_all_spline(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   int index_y,
		   int index_ddy,
		   double * result,
		   ErrorMsg errmsg) {

  int i;
  double h;

  *result = 0;

  for (i=0; i < n_lines-1; i++) {

    h = (array[(i+1)*n_columns+index_x]-array[i*n_columns+index_x]);

    *result +=
      (array[i*n_columns+index_y]+array[(i+1)*n_columns+index_y])*h/2.+
      (array[i*n_columns+index_ddy]+array[(i+1)*n_columns+index_ddy])*h*h*h/24.;

  }

  return _SUCCESS_;
}

int array_integrate_all_trapzd_or_spline(
		   double * array,
		   int n_columns,
		   int n_lines,
           int index_start_spline,
		   int index_x,   /** from 0 to (n_columns-1) */
		   int index_y,
		   int index_ddy,
		   double * result,
		   ErrorMsg errmsg) {

  int i;
  double h;

  if ((index_start_spline<0) || (index_start_spline>=n_lines)) {
    sprintf(errmsg,"%s(L:%d) index_start_spline outside of range",__func__,__LINE__);
    return _FAILURE_;
  }

  *result = 0;

  /* trapezoidal integration till given index */

  for (i=0; i < index_start_spline; i++) {

    h = (array[(i+1)*n_columns+index_x]-array[i*n_columns+index_x]);

    *result +=
      (array[i*n_columns+index_y]+array[(i+1)*n_columns+index_y])*h/2.;

  }

  /* then, spline integration */

  for (i=index_start_spline; i < n_lines-1; i++) {

    h = (array[(i+1)*n_columns+index_x]-array[i*n_columns+index_x]);

    *result +=
      (array[i*n_columns+index_y]+array[(i+1)*n_columns+index_y])*h/2.+
      (array[i*n_columns+index_ddy]+array[(i+1)*n_columns+index_ddy])*h*h*h/24.;

  }

  return _SUCCESS_;
}

 /**
 * Not called.
 */
int array_integrate(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   int index_y,
		   int index_int_y_dx,
		   ErrorMsg errmsg) {

  int i;
  double sum;

  if ((index_int_y_dx == index_x) || (index_int_y_dx == index_y)) {
    sprintf(errmsg,"%s(L:%d) : Output column %d must differ from input columns %d and %d",__func__,__LINE__,index_int_y_dx,index_x,index_y);
    return _FAILURE_;
  }

  sum=0.;
  *(array+0*n_columns+index_int_y_dx)=sum;

  for (i=1; i<n_lines; i++) {

    sum += 0.5 * (*(array+i*n_columns+index_y) + *(array+(i-1)*n_columns+index_y))
               * (*(array+i*n_columns+index_x) - *(array+(i-1)*n_columns+index_x));

    *(array+i*n_columns+index_int_y_dx)=sum;
  }


  return _SUCCESS_;
}

 /**
 * Called by thermodynamics_init().
 */
int array_integrate_ratio(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   int index_y1,
		   int index_y2,
		   int index_int_y1_over_y2_dx,
		   ErrorMsg errmsg) {

  int i;
  double sum;

  if ((index_int_y1_over_y2_dx == index_x) || (index_int_y1_over_y2_dx == index_y1) || (index_int_y1_over_y2_dx == index_y2)) {
    sprintf(errmsg,"%s(L:%d) : Output column %d must differ from input columns %d, %d and %d",__func__,__LINE__,index_int_y1_over_y2_dx,index_x,index_y1,index_y2);
    return _FAILURE_;
  }

  sum=0.;

  *(array+0*n_columns+index_int_y1_over_y2_dx)=sum;

  for (i=1; i<n_lines; i++) {

    sum += 0.5 * (*(array+i*n_columns+index_y1) / *(array+i*n_columns+index_y2)
		  + *(array+(i-1)*n_columns+index_y1) / *(array+(i-1)*n_columns+index_y2))
      * (*(array+i*n_columns+index_x) - *(array+(i-1)*n_columns+index_x));

    *(array+i*n_columns+index_int_y1_over_y2_dx)=sum;
  }


  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are all columns of the same array
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   double x,
		   int * last_index,
		   double * result,
		   int result_size, /** from 1 to n_columns */
		   ErrorMsg errmsg) {

  int inf,sup,mid,i;
  double weight;

  inf=0;
  sup=n_lines-1;

  if (*(array+inf*n_columns+index_x) < *(array+sup*n_columns+index_x)){

    if (x < *(array+inf*n_columns+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,*(array+inf*n_columns+index_x));
      return _FAILURE_;
    }

    if (x > *(array+sup*n_columns+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,*(array+sup*n_columns+index_x));
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < *(array+mid*n_columns+index_x)) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < *(array+sup*n_columns+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,*(array+sup*n_columns+index_x));
      return _FAILURE_;
    }

    if (x > *(array+inf*n_columns+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,*(array+inf*n_columns+index_x));
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > *(array+mid*n_columns+index_x)) {sup=mid;}
      else {inf=mid;}

    }

  }

  *last_index = inf;

  weight=(x-*(array+inf*n_columns+index_x))/(*(array+sup*n_columns+index_x)-*(array+inf*n_columns+index_x));

  for (i=0; i<result_size; i++)
    *(result+i) = *(array+inf*n_columns+i) * (1.-weight)
      + weight * *(array+sup*n_columns+i);

  *(result+index_x) = x;

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are in different arrays
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate_spline(
                             double * __restrict__ x_array,
                             int n_lines,
                             double * __restrict__ array,
                             double * __restrict__ array_splined,
                             int n_columns,
                             double x,
                             int * __restrict__ last_index,
                             double * __restrict__ result,
                             int result_size, /** from 1 to n_columns */
                             ErrorMsg errmsg) {

  int inf,sup,mid,i;
  double h,a,b;

  inf=0;
  sup=n_lines-1;

  if (x_array[inf] < x_array[sup]){

    if (x < x_array[inf]) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
      return _FAILURE_;
    }

    if (x > x_array[sup]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[sup]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < x_array[sup]) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[sup]);
      return _FAILURE_;
    }

    if (x > x_array[inf]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[inf]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  *last_index = inf;

  h = x_array[sup] - x_array[inf];
  b = (x-x_array[inf])/h;
  a = 1-b;

  for (i=0; i<result_size; i++)
    *(result+i) =
      a * *(array+inf*n_columns+i) +
      b * *(array+sup*n_columns+i) +
      ((a*a*a-a)* *(array_splined+inf*n_columns+i) +
       (b*b*b-b)* *(array_splined+sup*n_columns+i))*h*h/6.;

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are in different arrays
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate_linear(
			     double * x_array,
			     int n_lines,
			     double * array,
			     int n_columns,
			     double x,
			     int * last_index,
			     double * result,
			     int result_size, /** from 1 to n_columns */
			     ErrorMsg errmsg) {

  int inf,sup,mid,i;
  double h,a,b;

  inf=0;
  sup=n_lines-1;

  if (x_array[inf] < x_array[sup]){

    if (x < x_array[inf]) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
      return _FAILURE_;
    }

    if (x > x_array[sup]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[sup]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < x_array[sup]) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[sup]);
      return _FAILURE_;
    }

    if (x > x_array[inf]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[inf]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  *last_index = inf;

  h = x_array[sup] - x_array[inf];
  b = (x-x_array[inf])/h;
  a = 1-b;

  for (i=0; i<result_size; i++)
    *(result+i) =
      a * *(array+inf*n_columns+i) +
      b * *(array+sup*n_columns+i);

  return _SUCCESS_;
}


/**
 * interpolate to get y_i(x), when x and y_i are in different arrays
 *
 * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
 */
int array_interpolate_logspline(
							 double * x_array,
							 int n_lines,
							 double * array,
							 double * array_logsplined,
							 int n_columns,
							 double x,
							 int * last_index,
							 double * result,
							 int result_size, /** from 1 to n_columns */
							 ErrorMsg errmsg) {

	int inf,sup,mid,i;
	double h,a,b;

	inf=0;
	sup=n_lines-1;

	if (x_array[inf] < x_array[sup]){

		if (x < x_array[inf]) {
			sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
			return _FAILURE_;
		}

		if (x > x_array[sup]) {
			sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[sup]);
			return _FAILURE_;
		}

		while (sup-inf > 1) {

			mid=(int)(0.5*(inf+sup));
			if (x < x_array[mid]) {sup=mid;}
			else {inf=mid;}

		}

	}

	else {

		if (x < x_array[sup]) {
			sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[sup]);
			return _FAILURE_;
		}

		if (x > x_array[inf]) {
			sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[inf]);
			return _FAILURE_;
		}

		while (sup-inf > 1) {

			mid=(int)(0.5*(inf+sup));
			if (x > x_array[mid]) {sup=mid;}
			else {inf=mid;}

		}

	}

	*last_index = inf;

	h = log(x_array[sup]) - log(x_array[inf]);
	b = (log(x)-log(x_array[inf]))/h;
	a = 1-b;

	for (i=0; i<result_size; i++)
		*(result+i) = exp(
		a * log(array[inf*n_columns+i]) +
		b * log(array[sup*n_columns+i]) +
		((a*a*a-a)* array_logsplined[inf*n_columns+i] +
		 (b*b*b-b)* array_logsplined[sup*n_columns+i])*h*h/6.);

	return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are in different arrays
  *
  *
  */
int array_interpolate_spline_one_column(
					double * x_array,
					int x_size,
					double * y_array, /* array of size x_size*y_size with elements
							   y_array[index_y*x_size+index_x] */
					int y_size,
					int index_y,
					double * ddy_array, /* array of size x_size*y_size */
					double x,   /* input */
					double * y, /* output */
					ErrorMsg errmsg
					) {


  int inf,sup,mid;
  double h,a,b;

  inf=0;
  sup=x_size-1;

  if (x_array[inf] < x_array[sup]){

    if (x < x_array[inf]) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
      return _FAILURE_;
    }

    if (x > x_array[sup]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[sup]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < x_array[sup]) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[sup]);
      return _FAILURE_;
    }

    if (x > x_array[inf]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[inf]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > x_array[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  h = x_array[sup] - x_array[inf];
  b = (x-x_array[inf])/h;
  a = 1-b;

  *y =
    a * y_array[index_y * x_size + inf] +
    b * y_array[index_y * x_size + sup] +
    ((a*a*a-a)* ddy_array[index_y * x_size + inf] +
     (b*b*b-b)* ddy_array[index_y * x_size + sup])*h*h/6.;

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are in different arrays
  *
  *
  */
int array_interpolate_extrapolate_spline_one_column(
						    double * x_array,
						    int x_size,
						    double * y_array, /* array of size x_size*y_size with elements
									 y_array[index_y*x_size+index_x] */
						    int y_size,
						    int index_y,
						    double * ddy_array, /* array of size x_size*y_size */
						    double x,   /* input */
						    double * y, /* output */
						    ErrorMsg errmsg
						    ) {


  int inf,sup,mid;
  double h,a,b;

  if (x > x_array[x_size-2] || x < x_array[0]) {

    /*interpolate/extrapolate linearly y as a function of x*/

    h = x_array[x_size-1] - x_array[x_size-2];
    b = (x-x_array[x_size-2])/h;
    a = 1-b;

    *y = a * y_array[index_y * x_size + (x_size-2)] +
	     b * y_array[index_y * x_size + (x_size-1)];


  }

  else {

    /*interpolate y as a function of x with a spline*/

    inf=0;
    sup=x_size-1;

    if (x_array[inf] < x_array[sup]){

      if (x < x_array[inf]) {
	sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
	return _FAILURE_;
      }

      if (x > x_array[sup]) {
	sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[sup]);
	return _FAILURE_;
      }

      while (sup-inf > 1) {

	mid=(int)(0.5*(inf+sup));
	if (x < x_array[mid]) {sup=mid;}
	else {inf=mid;}

      }

    }

    else {

      if (x < x_array[sup]) {
	sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[sup]);
	return _FAILURE_;
      }

      if (x > x_array[inf]) {
	sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[inf]);
	return _FAILURE_;
      }

      while (sup-inf > 1) {

	mid=(int)(0.5*(inf+sup));
	if (x > x_array[mid]) {sup=mid;}
	else {inf=mid;}

      }

    }

    h = x_array[sup] - x_array[inf];
    b = (x-x_array[inf])/h;
    a = 1-b;

    *y =
      a * y_array[index_y * x_size + inf] +
      b * y_array[index_y * x_size + sup] +
      ((a*a*a-a)* ddy_array[index_y * x_size + inf] +
       (b*b*b-b)* ddy_array[index_y * x_size + sup])*h*h/6.;

  }

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are in different arrays
  *
  *
  */
int array_interpolate_extrapolate_logspline_loglinear_one_column(
								 double * x_array,
								 int x_size,
								 int x_stop,
								 double * y_array, /* array of size x_size*y_size with elements
										      y_array[index_y*x_size+index_x] */
								 int y_size,
								 int index_y,
								 double * ddlogy_array, /* array of size x_size*y_size */
								 double x,   /* input */
								 double * y, /* output */
								 ErrorMsg errmsg
								 ) {


  int inf,sup,mid;
  double h,a,b;

  if (x > x_array[x_stop-1]) {

    /*interpolate/extrapolate linearly ln(y) as a function of ln(x)*/

    h = log(x_array[x_stop-1]) - log(x_array[x_stop-2]);
    b = (log(x)-log(x_array[x_stop-2]))/h;
    a = 1-b;

/*     *y = exp(a * log(y_array[index_y * x_size + (x_stop-2)]) + */
/* 	     b * log(y_array[index_y * x_size + (x_stop-1)])); */

    *y = exp(log(y_array[index_y * x_size + (x_stop-1)])
	     +(log(x)-log(x_array[x_stop-1]))
	     *((log(y_array[index_y * x_size + (x_stop-1)])-log(y_array[index_y * x_size + (x_stop-2)]))/h
	       +h/6.*(ddlogy_array[index_y * x_size + (x_stop-2)]+2.*ddlogy_array[index_y * x_size + (x_stop-1)])));


  }

  else {

    /*interpolate ln(y) as a function of ln(x) with a spline*/

    inf=0;
    sup=x_stop-1;

    if (x_array[inf] < x_array[sup]){

      if (x < x_array[inf]) {
	sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
	return _FAILURE_;
      }

      if (x > x_array[sup]) {
	sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[sup]);
	return _FAILURE_;
      }

      while (sup-inf > 1) {

	mid=(int)(0.5*(inf+sup));
	if (x < x_array[mid]) {sup=mid;}
	else {inf=mid;}

      }

    }

    else {

      if (x < x_array[sup]) {
	sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[sup]);
	return _FAILURE_;
      }

      if (x > x_array[inf]) {
	sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,x_array[inf]);
	return _FAILURE_;
      }

      while (sup-inf > 1) {

	mid=(int)(0.5*(inf+sup));
	if (x > x_array[mid]) {sup=mid;}
	else {inf=mid;}

      }

    }

    h = log(x_array[sup]) - log(x_array[inf]);
    b = (log(x)-log(x_array[inf]))/h;
    a = 1-b;

    *y = exp(a * log(y_array[index_y * x_size + inf]) +
	     b * log(y_array[index_y * x_size + sup]) +
	     ((a*a*a-a)* ddlogy_array[index_y * x_size + inf] +
	      (b*b*b-b)* ddlogy_array[index_y * x_size + sup])*h*h/6.);

  }

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are all columns of the same array, x is arranged in growing order, and the point x is presumably close to the previous point x from the last call of this function.
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate_growing_closeby(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   double x,
		   int * last_index,
		   double * result,
		   int result_size, /** from 1 to n_columns */
		   ErrorMsg errmsg) {

  int inf,sup,i;
  double weight;

  inf = *last_index;
  sup = *last_index+1;

  while (x < *(array+inf*n_columns+index_x)) {
    inf--;
    if (inf < 0) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,
	      x,array[index_x]);
      return _FAILURE_;
    }
  }
  sup = inf+1;
  while (x > *(array+sup*n_columns+index_x)) {
    sup++;
    if (sup > (n_lines-1)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,
	      x,array[(n_lines-1)*n_columns+index_x]);
      return _FAILURE_;
    }
  }
  inf = sup-1;

  *last_index = inf;

  weight=(x-*(array+inf*n_columns+index_x))/(*(array+sup*n_columns+index_x)-*(array+inf*n_columns+index_x));

  for (i=0; i<result_size; i++)
    *(result+i) = *(array+inf*n_columns+i) * (1.-weight)
      + weight * *(array+sup*n_columns+i);

  *(result+index_x) = x;

  return _SUCCESS_;
}

/**
  * interpolate to get y(x), when x and y are two columns of the same array, x is arranged in growing order, and the point x is presumably close to the previous point x from the last call of this function.
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate_one_growing_closeby(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   double x,
		   int * last_index,
           int index_y,
		   double * result,
		   ErrorMsg errmsg) {

  int inf,sup;
  double weight;

  inf = *last_index;
  sup = *last_index+1;

  while (x < *(array+inf*n_columns+index_x)) {
    inf--;
    if (inf < 0) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,
	      x,array[index_x]);
      return _FAILURE_;
    }
  }
  sup = inf+1;
  while (x > *(array+sup*n_columns+index_x)) {
    sup++;
    if (sup > (n_lines-1)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,
	      x,array[(n_lines-1)*n_columns+index_x]);
      return _FAILURE_;
    }
  }
  inf = sup-1;

  *last_index = inf;

  weight=(x-*(array+inf*n_columns+index_x))/(*(array+sup*n_columns+index_x)-*(array+inf*n_columns+index_x));

  *result = *(array+inf*n_columns+index_y) * (1.-weight) + *(array+sup*n_columns+index_y) * weight;

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are all columns of the same array, x is arranged in growing order, and the point x is presumably very close to the previous point x from the last call of this function.
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate_spline_growing_closeby(
					     double * x_array,
					     int n_lines,
					     double * array,
					     double * array_splined,
					     int n_columns,
					     double x,
					     int * last_index,
					     double * result,
					     int result_size, /** from 1 to n_columns */
					     ErrorMsg errmsg) {

  int inf,sup,i;
  double h,a,b;

  /*
  if (*last_index < 0) {
    sprintf(errmsg,"%s(L:%d) problem with last_index =%d < 0",__func__,__LINE__,*last_index);
    return _FAILURE_;
  }
  if (*last_index > (n_lines-1)) {
    sprintf(errmsg,"%s(L:%d) problem with last_index =%d > %d",__func__,__LINE__,*last_index,n_lines-1);
    return _FAILURE_;
  }
  */

  inf = *last_index;
  class_test(inf<0 || inf>(n_lines-1),
	     errmsg,
	     "*lastindex=%d out of range [0:%d]\n",inf,n_lines-1);
  while (x < x_array[inf]) {
    inf--;
    if (inf < 0) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,
	      x,x_array[0]);
      return _FAILURE_;
    }
  }
  sup = inf+1;
  while (x > x_array[sup]) {
    sup++;
    if (sup > (n_lines-1)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,
	      x,x_array[n_lines-1]);
      return _FAILURE_;
    }
  }
  inf = sup-1;

  *last_index = inf;

  h = x_array[sup] - x_array[inf];
  b = (x-x_array[inf])/h;
  a = 1-b;

  for (i=0; i<result_size; i++)
    *(result+i) =
      a * *(array+inf*n_columns+i) +
      b * *(array+sup*n_columns+i) +
      ((a*a*a-a)* *(array_splined+inf*n_columns+i) +
       (b*b*b-b)* *(array_splined+sup*n_columns+i))*h*h/6.;

  return _SUCCESS_;
}

 /**
  * interpolate to get y_i(x), when x and y_i are all columns of the same array, x is arranged in growing order, and the point x is presumably close (but maybe not so close) to the previous point x from the last call of this function.
  *
  * Called by background_at_eta(); background_eta_of_z(); background_solve(); thermodynamics_at_z().
  */
int array_interpolate_spline_growing_hunt(
					     double * x_array,
					     int n_lines,
					     double * array,
					     double * array_splined,
					     int n_columns,
					     double x,
					     int * last_index,
					     double * result,
					     int result_size, /** from 1 to n_columns */
					     ErrorMsg errmsg) {

  int inf,sup,mid,i,inc;
  double h,a,b;

  inc=1;

  if (x >= x_array[*last_index]) {
    if (x > x_array[n_lines-1]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,
	      x,x_array[n_lines-1]);
      return _FAILURE_;
    }
    /* try closest neighboor upward */
    inf = *last_index;
    sup = inf + inc;
    if (x > x_array[sup]) {
      /* hunt upward */
      while (x > x_array[sup]) {
	inf = sup;
	inc += 1;
	sup += inc;
	if (sup > n_lines-1) {
	  sup = n_lines-1;
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
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,
	      x,x_array[0]);
      return _FAILURE_;
    }
    /* try closest neighboor downward */
    sup = *last_index;
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

  *last_index = inf;

  h = x_array[sup] - x_array[inf];
  b = (x-x_array[inf])/h;
  a = 1-b;

  for (i=0; i<result_size; i++)
    *(result+i) =
      a * *(array+inf*n_columns+i) +
      b * *(array+sup*n_columns+i) +
      ((a*a*a-a)* *(array_splined+inf*n_columns+i) +
       (b*b*b-b)* *(array_splined+sup*n_columns+i))*h*h/6.;

  return _SUCCESS_;
}

/**
 * interpolate linearily to get y_i(x), when x and y_i are in two different arrays
 *
 * Called by transfer_interpolate_sources(); transfer_functions_at_k(); perturb_sources_at_eta().
 */
int array_interpolate_two(
		   double * array_x,
		   int n_columns_x,
		   int index_x,   /** from 0 to (n_columns_x-1) */
		   double * array_y,
		   int n_columns_y,
		   int n_lines,  /** must be the same for array_x and array_y */
		   double x,
		   double * result,
		   int result_size, /** from 1 to n_columns_y */
		   ErrorMsg errmsg) {

  int inf,sup,mid,i;
  double weight;

  inf=0;
  sup=n_lines-1;

  if (array_x[inf*n_columns_x+index_x] < array_x[sup*n_columns_x+index_x]){

    if (x < array_x[inf*n_columns_x+index_x]) {

      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,array_x[inf*n_columns_x+index_x]);
      return _FAILURE_;
    }

    if (x > array_x[sup*n_columns_x+index_x]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,array_x[sup*n_columns_x+index_x]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < array_x[mid*n_columns_x+index_x]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < *(array_x+sup*n_columns_x+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,*(array_x+sup*n_columns_x+index_x));
      return _FAILURE_;
    }

    if (x > *(array_x+inf*n_columns_x+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,*(array_x+inf*n_columns_x+index_x));
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > *(array_x+mid*n_columns_x+index_x)) {sup=mid;}
      else {inf=mid;}

    }

  }

  weight=(x-*(array_x+inf*n_columns_x+index_x))/(*(array_x+sup*n_columns_x+index_x)-*(array_x+inf*n_columns_x+index_x));

  for (i=0; i<result_size; i++)
    *(result+i) = *(array_y+i*n_lines+inf) * (1.-weight)
      + weight * *(array_y+i*n_lines+sup) ;

  return _SUCCESS_;
}

/**
 * Same as array_interpolate_two, but with order of indices exchanged in array_y
 */
int array_interpolate_two_bis(
		   double * array_x,
		   int n_columns_x,
		   int index_x,   /** from 0 to (n_columns_x-1) */
		   double * array_y,
		   int n_columns_y,
		   int n_lines,  /** must be the same for array_x and array_y */
		   double x,
		   double * result,
		   int result_size, /** from 1 to n_columns_y */
		   ErrorMsg errmsg) {

  int inf,sup,mid,i;
  double weight;

  inf=0;
  sup=n_lines-1;

  if (array_x[inf*n_columns_x+index_x] < array_x[sup*n_columns_x+index_x]){

    if (x < array_x[inf*n_columns_x+index_x]) {

      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,array_x[inf*n_columns_x+index_x]);
      return _FAILURE_;
    }

    if (x > array_x[sup*n_columns_x+index_x]) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,array_x[sup*n_columns_x+index_x]);
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < array_x[mid*n_columns_x+index_x]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    if (x < *(array_x+sup*n_columns_x+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,*(array_x+sup*n_columns_x+index_x));
      return _FAILURE_;
    }

    if (x > *(array_x+inf*n_columns_x+index_x)) {
      sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,x,*(array_x+inf*n_columns_x+index_x));
      return _FAILURE_;
    }

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > *(array_x+mid*n_columns_x+index_x)) {sup=mid;}
      else {inf=mid;}

    }

  }

  weight=(x-*(array_x+inf*n_columns_x+index_x))/(*(array_x+sup*n_columns_x+index_x)-*(array_x+inf*n_columns_x+index_x));

  for (i=0; i<result_size; i++)
    *(result+i) = *(array_y+inf*n_columns_y+i) * (1.-weight)
      + weight * *(array_y+sup*n_columns_y+i) ;

  return _SUCCESS_;
}


/**
 * interpolate linearily to get y_i(x), when x and y_i are in two different arrays
 *
 * Called by transfer_interpolate_sources(); transfer_functions_at_k(); perturb_sources_at_eta().
 */
int array_interpolate_two_arrays_one_column(
					    double * array_x, /* assumed to be a vector (i.e. one column array) */
					    double * array_y,
					    int n_columns_y,
					    int index_y, /* between 0 and (n_columns_y-1) */
					    int n_lines,  /** must be the same for array_x and array_y */
					    double x,
					    double * result,
					    ErrorMsg errmsg) {

  int inf,sup,mid;
  double weight;

  inf=0;
  sup=n_lines-1;

  if (array_x[inf] < array_x[sup]){

    class_test(x < array_x[inf],
	       errmsg,
	       "x=%e < x_min=%e",x,array_x[inf]);

    class_test(x > array_x[sup],
	       errmsg,
	       "x=%e > x_max=%e",x,array_x[sup]);

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x < array_x[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  else {

    class_test(x < array_x[sup],
	       errmsg,
	       "x=%e < x_min=%e",x,array_x[sup]);

    class_test(x > array_x[inf],
	       errmsg,
	       "x=%e > x_max=%e",x,array_x[inf]);

    while (sup-inf > 1) {

      mid=(int)(0.5*(inf+sup));
      if (x > array_x[mid]) {sup=mid;}
      else {inf=mid;}

    }

  }

  weight=(x-array_x[inf])/(array_x[sup]-array_x[inf]);

  *result = array_y[index_y*n_lines+inf] * (1.-weight)
    + weight * array_y[index_y*n_lines+sup];

  return _SUCCESS_;
}

/**
 * Called by transfer_solve().
 */
int array_interpolate_equal(
			    double * array,
			    int n_columns,
			    int n_lines,
			    double x,
			    double x_min,
			    double x_max,
			    double * result,
			    ErrorMsg errmsg) {

  int index_minus,i;
  double x_step,x_minus,weight;

  if (x < x_min) {
    sprintf(errmsg,"%s(L:%d) : x out of bounds: x=%e,x_min=%e",__func__,__LINE__,x,x_min);
    return _FAILURE_;
  }

  if (x > x_max) {
    sprintf(errmsg,"%s(L:%d) : x out of bounds: x=%e,x_max=%e",__func__,__LINE__,x,x_max);
    return _FAILURE_;
  }

  x_step = (x_max-x_min)/(n_lines-1);
  index_minus = (int)((x-x_min)/x_step);
  x_minus = index_minus * x_step;
  weight = (x-x_minus) / x_step;

  for (i=0; i<n_columns; i++)
    result[i] = *(array+n_columns*index_minus+i)*(1.-weight)
      + *(array+n_columns*(index_minus+1)+i)*weight;

  return _SUCCESS_;

}

/**
 * cubic interpolation of array with equally space abscisses
 */

int array_interpolate_cubic_equal(
				  double x0,
				  double dx,
				  double *yarray,
				  int Nx,
				  double x,
				  double * result,
				  ErrorMsg errmsg) {

  int i;
  double frac;

  class_test((dx > 0 && (x<x0 || x>x0+dx*(Nx-1))),
	     errmsg,
	     "x=%e out of range [%e %e]",x,x0,x0+dx*(Nx-1));

  class_test((dx < 0 && (x>x0 || x<x0+dx*(Nx-1))),
	     errmsg,
	     "x=%e out of range [%e %e]",x,x0+dx*(Nx-1),x0);

  i = (int)floor((x-x0)/dx);
  if (i<1) i=1;
  if (i>Nx-3) i=Nx-3;
  frac = (x-x0)/dx-i;
  yarray += i-1;

  *result=-yarray[0]*frac*(1.-frac)*(2.-frac)/6.
    +yarray[1]*(1.+frac)*(1.-frac)*(2.-frac)/2.
    +yarray[2]*(1.+frac)*frac*(2.-frac)/2.
    +yarray[3]*(1.+frac)*frac*(frac-1.)/6.;

  return _SUCCESS_;
}

int array_interpolate_parabola(double x1,
			       double x2,
			       double x3,
			       double x,
			       double y1,
			       double y2,
			       double y3,
			       double * y,
			       double * dy,
			       double * ddy,
			       ErrorMsg errmsg) {

  double a,b,c;

  /*
    a x_i**2 + b x_i + c = y_i

    a (x1**2-x2**2) + b (x1-x2) = y1-y2
    a (x3**2-x2**2) + b (x3-x2) = y3-y2

    a (x1**2-x2**2)(x3**2-x2**2) + b (x1-x2)(x3**2-x2**2) = (y1-y2)(x3**2-x2**2)
    a (x3**2-x2**2)(x1**2-x2**2) + b (x3-x2)(x1**2-x2**2) = (y3-y2)(x1**2-x2**2)

    b = [(y1-y2)(x3**2-x2**2) - (y3-y2)(x1**2-x2**2)]/(x1-x2)(x3-x2)(x3-x1)

  */

  b = ((y1-y2)*(x3-x2)*(x3+x2) - (y3-y2)*(x1-x2)*(x1+x2))/(x1-x2)/(x3-x2)/(x3-x1);

  a = (y1-y2-b*(x1-x2))/(x1-x2)/(x1+x2);

  c = y2 - b*x2 - a*x2*x2;

  *y = a*x*x + b*x + c;
  *dy = 2.*a*x + b;
  *ddy = 2.*a;

  return _SUCCESS_;

}

/**
 * Called by transfer_solve().
 */
int array_integrate_all(
		   double * array,
		   int n_columns,
		   int n_lines,
		   int index_x,   /** from 0 to (n_columns-1) */
		   int index_y,
		   double *result) {

  int i;
  double sum;

  sum=0.;

  for (i=1; i<n_lines; i++) {

    sum += 0.5 * (*(array+i*n_columns+index_y) + *(array+(i-1)*n_columns+index_y))
               * (*(array+i*n_columns+index_x) - *(array+(i-1)*n_columns+index_x));

  }

  *result = sum;

  return _SUCCESS_;

}

int array_smooth_trg(double * array,
		     int k_size,
		     int starting_k,
		     int eta_size,
		     int index_eta,
		     int radius, /*3, 5 or 7 */
		     ErrorMsg errmsg) {

  double * smooth;
  int i,j,jmin,jmax;
  double weigth;
  double *coeff;

  smooth=malloc(k_size*sizeof(double));
  if (smooth == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate smooth",__func__,__LINE__);
    return _FAILURE_;
  }

  class_calloc(coeff,2*radius+1,sizeof(double),errmsg);

  switch(radius){
  case 3:
    weigth = 21;

    coeff[0] = -2;
    coeff[1] = 3;
    coeff[2] = 6;
    coeff[3] = 7;
    coeff[4] = 6;
    coeff[5] = 3;
    coeff[6] = -2;

    break;
  case 4:
    weigth = 231;

    coeff[0] = -21;
    coeff[1] = 14;
    coeff[2] = 39;
    coeff[3] = 54;
    coeff[4] = 59;
    coeff[5] = 54;
    coeff[6] = 39;
    coeff[7] = 14;
    coeff[8] = -21;

    break;
  case 5:
    weigth = 429;

    coeff[0] = -36;
    coeff[1] = 9;
    coeff[2] = 44;
    coeff[3] = 69;
    coeff[4] = 84;
    coeff[5] = 89;
    coeff[6] = 84;
    coeff[7] = 69;
    coeff[8] = 44;
    coeff[9] = 9;
    coeff[10] = -36;

    break;
  case 6:
    weigth = 143;

    coeff[0] = -11;
    coeff[1] = 0;
    coeff[2] = 9;
    coeff[3] = 16;
    coeff[4] = 21;
    coeff[5] = 24;
    coeff[6] = 25;
    coeff[7] = 24;
    coeff[8] = 21;
    coeff[9] = 16;
    coeff[10] = 9;
    coeff[11] = 0;
    coeff[12] = -11;

    break;
  case 7:
    weigth = 1105;

    coeff[0] = -78;
    coeff[1] = -13;
    coeff[2] = 42;
    coeff[3] = 87;
    coeff[4] = 122;
    coeff[5] = 147;
    coeff[6] = 162;
    coeff[7] = 167;
    coeff[8] = 162;
    coeff[9] = 147;
    coeff[10] = 122;
    coeff[11] = 87;
    coeff[12] = 42;
    coeff[13] = -13;
    coeff[14] = -78;

    break;

/*   case 8: */


  default:
    class_stop(errmsg,"Non valid radius %d: please chose between 3 4 5 or 6\n",radius);
    weigth=0;
    break;
  }

  for (i=starting_k; i<k_size-radius; i++) {
      smooth[i]=0.;
      jmin = MAX(i-radius,0);
      jmax = MIN(i+radius,k_size-1);
      for (j=jmin; j <= jmax; j++) {
	smooth[i] += coeff[j-jmin]*array[j+k_size*index_eta];
      }
      smooth[i] /= weigth;
  }

  for (i=starting_k; i<k_size-radius; i++)
    array[i+k_size*index_eta] = smooth[i];

  free(smooth);
  free(coeff);

  return _SUCCESS_;

}

int array_smooth(double * array,
		 int n_columns,
		 int n_lines,
		 int index, /** from 0 to (n_columns-1) */
		 int radius,
		 ErrorMsg errmsg) {

  double * smooth;
  int i,j,jmin,jmax;
  double weigth;

  smooth=malloc(n_lines*sizeof(double));
  if (smooth == NULL) {
    sprintf(errmsg,"%s(L:%d) Cannot allocate smooth",__func__,__LINE__);
    return _FAILURE_;
  }

  for (i=0; i<n_lines; i++) {
    smooth[i]=0.;
    weigth=0.;
    jmin = MAX(i-radius,0);
    jmax = MIN(i+radius,n_lines-1);
    for (j=jmin; j <= jmax; j++) {
      smooth[i] += array[j*n_columns+index];
      weigth += 1.;
    }
    smooth[i] /= weigth;
  }

  for (i=0; i<n_lines; i++)
    array[i*n_columns+index] = smooth[i];

  free(smooth);

  return _SUCCESS_;

}

/**
 * Compute quadrature weights for the trapezoidal integration method, xhen x is in gorwing order.
 *
 * @param x                     Input: Grid points on which f() is known.
 * @param n                     Input: number of grid points.
 * @param w_trapz               Output: Weights of the trapezoidal method.
 * @return the error status
 */

int array_trapezoidal_weights(
                              double * __restrict__ x,
                              int n,
                              double * __restrict__ w_trapz,
                              ErrorMsg errmsg
                              ) {
  int i;

  /* Case with just one point, w would normally be 0. */
  if (n==1){
    w_trapz[0] = 0.0;
  }
  else if (n>1){
    //Set edgeweights:
    w_trapz[0] = 0.5*(x[1]-x[0]);
    w_trapz[n-1] = 0.5*(x[n-1]-x[n-2]);
    //Set inner weights:
    for (i=1; i<(n-1); i++){
      w_trapz[i] = 0.5*(x[i+1]-x[i-1]);
    }
  }
  return _SUCCESS_;
}

/**
 * Compute quadrature weights for the trapezoidal integration method, when x is in decreasing order.
 *
 * @param x                     Input: Grid points on which f() is known.
 * @param n                     Input: number of grid points.
 * @param w_trapz               Output: Weights of the trapezoidal method.
 * @return the error status
 */

int array_trapezoidal_mweights(
                              double * __restrict__ x,
                              int n,
                              double * __restrict__ w_trapz,
                              ErrorMsg errmsg
                              ) {
  int i;

  /* Case with just one point. */
  if (n==1){
    w_trapz[0] = 1.0;
  }
  else if (n>1){
    //Set edgeweights:
    w_trapz[0] = 0.5*(x[0]-x[1]);
    w_trapz[n-1] = 0.5*(x[n-2]-x[n-1]);
    //Set inner weights:
    for (i=1; i<(n-1); i++){
      w_trapz[i] = 0.5*(x[i-1]-x[i+1]);
    }
  }
  return _SUCCESS_;
}

/**
 * Compute integral of function using trapezoidal method.
 *
 * @param integrand             Input: The function we are integrating.
 * @param n                     Input: Compute integral on grid [0;n-1].
 * @param w_trapz               Input: Weights of the trapezoidal method.
 * @param I                     Output: The integral.
 * @return the error status
 */

int array_trapezoidal_integral(
                                  double * __restrict__ integrand,
                                  int n,
                                  double * __restrict__ w_trapz,
                                  double * __restrict__ I,
                                  ErrorMsg errmsg
                                  ) {
  int i;
  double res=0.0;
  for (i=0; i<n; i++){
    res += integrand[i]*w_trapz[i];
  }
  *I = res;
  return _SUCCESS_;
}

/**
 * Compute convolution integral of product of two functions using trapezoidal method.
 *
 * @param integrand1            Input: Function 1.
 * @param integrand2            Input: Function 2.
 * @param n                     Input: Compute integral on grid [0;n-1].
 * @param w_trapz               Input: Weights of the trapezoidal method.
 * @param I                     Output: The integral.
 * @return the error status
 */

int array_trapezoidal_convolution(
                                     double * __restrict__ integrand1,
                                     double * __restrict__ integrand2,
                                     int n,
                                     double * __restrict__ w_trapz,
                                     double * __restrict__ I,
                                     ErrorMsg errmsg
                                     ) {
  int i;
  double res=0.0;
  for (i=0; i<n; i++){
    res += integrand1[i]*integrand2[i]*w_trapz[i];
  }
  *I = res;
  return _SUCCESS_;
}




/**
 * Compute weights and absiccas for Gaussian quadrature rules, including
 * Gauss-Chebyshev integration (both versions)
 * Gauss-Legendre integration
 * Gauss-Legendre half-integration
 * and normal trapezoidal integration
 *
 * See the arrays.h file for their descriptions, or open the manual.
 *
 * @param xarray                Input: Allocated array of size N, in which absiccas are stored
 * @param warray                Input: Allocated array of size N, in which weights are stored
 * @param N                     Input: Array size
 * @param gauss_type            Input: Method to use for integration
 * @return the error status
 */

int array_weights_gauss(double* xarray, double* warray, int N,short gauss_type,ErrorMsg err_msg){
  int i;
  class_test(N<1,
             err_msg,
             "invalid array size for integration");
  if(N==1){*xarray=0.0; *warray = 2.;return _SUCCESS_;}
  if(gauss_type==gauss_type_chebyshev_1){
    for(i=0;i<N;++i){
      xarray[i]=cos((2.0*i+1.0)/(2.0*(double)N)*_PI_);
      warray[i]=_PI_/(double)N;
    }
    return _SUCCESS_;
  }
  else if(gauss_type==gauss_type_chebyshev_2){
    for(i=0;i<N;++i){
      xarray[i]=cos((i+1.0)/((double)N+1.0)*_PI_);
      double sinval = sin((i+1.0)/((double)N+1.0)*_PI_);
      warray[i]=_PI_/((double)N+1.0)*sinval*sinval;
    }
    return _SUCCESS_;
  }
  else if(gauss_type==gauss_type_legendre){
    int Nhalf,j;
    double zero,zeroprev,pol,dpol,polprev,polnext;

    Nhalf = 0.5*((double)(N+1));
    for(i=0;i<Nhalf;++i){
      zero = cos(_PI_*(2.0*i+1.5)/(2.0*(double)N+1.0));
      do{
        polnext = 1.0;
        pol = 0.0;
        for(j=1;j<=N;++j){
          polprev = pol;
          pol = polnext;
          polnext = ((2.0*j-1.0)*zero*pol-(j-1.0)*polprev)/j;
        }
        dpol = N*(zero*polnext-pol)/(zero*zero-1.0);
        zeroprev = zero;
        zero = zeroprev-polnext/dpol;
      }while(fabs(zero-zeroprev)>GAUSS_EPSILON);
      xarray[i]=-zero;
      xarray[N-1-i]=zero;
      warray[i]=2.0/((1.0-zero*zero)*dpol*dpol);
      warray[N-1-i]=warray[i];
    }
    return _SUCCESS_;
  }
  else if(gauss_type==gauss_type_legendre_half){
    int Nhalf,j;
    double zero,zeroprev,pol,dpol,polprev,polnext;

    Nhalf = 0.5*((double)((2*N-1)+1));
    for(i=0;i<Nhalf;++i){
      zero = cos(_PI_*(2.0*i+1.5)/(2.0*(double)(2*N-1)+1.0));
      do{
        polnext = 1.0;
        pol = 0.0;
        for(j=1;j<=(2*N-1);++j){
          polprev = pol;
          pol = polnext;
          polnext = ((2.0*j-1.0)*zero*pol-(j-1.0)*polprev)/j;
        }
        dpol = (2*N-1)*(zero*polnext-pol)/(zero*zero-1.0);
        zeroprev = zero;
        zero = zeroprev-polnext/dpol;
      }while(fabs(zero-zeroprev)>GAUSS_EPSILON);
      xarray[N-1-i]=2*zero-1;
      warray[N-1-i]=4.0/((1.0-zero*zero)*dpol*dpol);
    }
    return _SUCCESS_;
  }
  /* This gauss-hermite method currently only works until N=150 due
   * to numerical underflow of the weights giving NaN's otherwise.
   * It will stay commented out for now.
   * TODO :: Find a way of extending beyond N=150
   * */
  /*else if(gauss_type==gauss_type_hermite){
    //Careful, has to go from -inf to inf, not -1 to 1
    int j,newstep,Nhalf;
    double polprev,pol,polnext,dpol,zero,zeroprev;
    Nhalf = 0.5*((double)N+1.0);
    class_test(N>150,err_msg,"Gauss-Hermite quadrature is not numerically viable with this huge N");
    for(i=1;i<=Nhalf;++i){
      if(i==1){
        zero = sqrt(2.0*N+1.0)-1.85575*pow(2.0*N+1.0,-1.0/6.0);
      }
      else if(i==2){
        zero -= 1.14*pow((double)N,0.426)/zero;
      }
      else if(i==3){
        zero = 1.86*zero-0.86*xarray[0];
      }
      else if(i==4){
        zero = 1.91*zero-0.91*xarray[1];
      }
      else{
        zero = 2.0*zero-xarray[i-3];
      }
      for(newstep=1;newstep<=GAUSS_HERMITE_MAXITER;++newstep){
        polnext = 0.25*_PI_;
        pol = 0.0;
        for(j=1;j<=N;++j){
          polprev = pol;
          pol = polprev;
          polnext = zero*sqrt(2.0/j)*pol-sqrt(((double)(j-1.0))/j)*polprev;
        }
        dpol = sqrt(2.0*N)*pol;
        zeroprev = zero;
        zero = zeroprev - polnext/dpol;
        if(fabs(zero-zeroprev)<=GAUSS_EPSILON){break;}
      }
      class_test((newstep>=GAUSS_HERMITE_MAXITER),
          err_msg,
          "no convergence of Newton's method during finding integration points for Gauss Hermite Quadrature."
          );
      xarray[i-1]=zero;
      xarray[N-i]=-zero;
      warray[i-1]=2.0/(dpol*dpol);
      warray[N-i]=warray[i-1];
    }
    return _SUCCESS_;
  }*/
  else if(gauss_type==gauss_type_trapezoid){
    for(i=0;i<N;++i){
      xarray[i] = 2.0*((double)i/(double)(N-1))-1.0;
      warray[i] = 2.0/(double)(N-1);
    }
    warray[0]*=0.5;
    warray[N-1]*=0.5;
    return _SUCCESS_;
  }
  else{
    class_stop(err_msg,
               "gauss integration type not recognized");
  }
}
/**
 * Compute weights and absiccas for Gaussian quadrature rules, including
 * Gauss-Chebyshev integration (both versions)
 * Gauss-Legendre integration
 * and normal trapezoidal integration
 *
 * Additionally rescales the weights and abscissas to conform to a
 * given integration region [xmin,xmax]
 *
 * @param xarray                Input: Allocated array of size N, in which absiccas are stored
 * @param warray                Input: Allocated array of size N, in which weights are stored
 * @param xmin                  Input: Minimum integration boundary
 * @param xmax                  Input: Maximum integration boundary
 * @param N                     Input: Array size
 * @param gauss_type            Input: Method to use for integration
 * @return the error status
 */
int array_weights_gauss_limits(double* xarray, double* warray,double xmin,double xmax, int N,short gauss_type,ErrorMsg err_msg){
  int i;
  /* Obtain absiccas, weights in the -1 to 1 region */
  array_weights_gauss(xarray,warray,N,gauss_type,err_msg);
  /* Rescale absiccas, weights to the xmin,xmax region */
  double xmean = 0.5*(xmax+xmin);
  double xdelta = 0.5*(xmax-xmin);
  for(i=1;i<N-1;++i){
    xarray[i]=xdelta*xarray[i]+xmean;
    warray[i]*=xdelta;
  }
  /* The first and last elements need to be precise */
  xarray[0] = xmin;
  warray[0]*=xdelta;
  xarray[N-1] = xmax;
  warray[N-1]*=xdelta;
  return _SUCCESS_;
}
/**
 * Rescales the weights and abscissas of a given
 * Gauss-integration to conform to another
 * integration region [xmin,xmax]
 *
 * This method allows the user to only compute the Gauss-weights once,
 * and subsequently rescale for any required integration with the same
 * number of total points N
 *
 * @param xarray                Input: Previous absiccas
 * @param warray                Input: Previous weights
 * @param xarray                Input: Allocated array of size N, in which final absiccas are stored
 * @param warray                Input: Allocated array of size N, in which final weights are stored
 * @param xmin                  Input: Minimum integration boundary
 * @param xmax                  Input: Maximum integration boundary
 * @param N                     Input: Array size
 * @return the error status
 */
int array_weights_gauss_rescale_limits(double* xarray,double* warray,double* xarrayres,double* warrayres,double xmin,double xmax,int N,ErrorMsg err_msg){
  int i;
  double xmean = 0.5*(xmax+xmin);
  double xdelta = 0.5*(xmax-xmin);
  for(i=0;i<N;++i){
    xarrayres[i]=xdelta*xarray[i]+xmean;
    warrayres[i]=warray[i]*xdelta;
  }
  return _SUCCESS_;
}
