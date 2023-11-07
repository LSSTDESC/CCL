#include <stdlib.h>
#include <math.h>

#include <complex.h>
#include <fftw3.h>
#include <stdio.h>

#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_sf_gamma.h>
#include "ccl.h"


/****************************************************************

This is the famous FFTLog.

First imlplemented by the living legend Andrew Hamilton:

http://casa.colorado.edu/~ajsh/FFTLog/

This version is a C version that was adapted from the C++ version found
in Copter JWG Carlson, another big loss for the cosmology community.

https://github.com/jwgcarlson/Copter

I've transformed this from C++ to C99 as the lowest common denominator
and provided bindings for C++ and python.

These are the C++ bindings

Modified by Paul Rogozenski (paulrogozenski@arizona.edu)
to implement a modified generalized version of FFTLog (https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf)
allowing computations of integrals over spherical bessel functions 
with any derivative of a (spherical) bessel function
and general power-law dependence of (kr)

relevant functions for generalized FFTLog:
  goodkr_new_deriv
  compute_u_coefficients_new_deriv
  general_fht
  ccl_fftlog_ComputeXi_general  


*****************************************************************/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.69314718056
#endif
/* This code is FFTLog, which is described in arXiv:astro-ph/9905191 */

static double complex lngamma_fftlog(double complex z)
{
  gsl_sf_result lnr, phi;
  gsl_sf_lngamma_complex_e(creal(z), cimag(z), &lnr, &phi);
  return lnr.val + I*phi.val;
}

static double complex polar (double r, double phi)
{
  return (r*cos(phi) +I*(r*sin(phi)));
}

static void lngamma_4(double x, double y, double* lnr, double* arg)
{
  double complex w;
  w = lngamma_fftlog(x+y*I);
  if(lnr) *lnr = creal(w);
  if(arg) *arg = cimag(w);
}

static double goodkr(int N, double mu, double q, double L, double kr)
{
  double xp = (mu+1+q)/2;
  double xm = (mu+1-q)/2;
  double y = M_PI*N/(2*L);
  double lnr, argm, argp;
  lngamma_4(xp, y, &lnr, &argp);
  lngamma_4(xm, y, &lnr, &argm);
  double arg = log(2/kr) * N/L + (argp + argm)/M_PI;
  double iarg = round(arg);
  if(arg != iarg)
    kr *= exp((arg - iarg)*L/N);
  return kr;
}


/* Pre-compute the coefficients that appear in the FFTLog implementation of
 * the discrete Hankel transform.  The parameters N, mu, and q here are the
 * same as for the function fht().  The parameter L is defined (for whatever
 * reason) to be N times the logarithmic spacing of the input array, i.e.
 *   L = N * log(r[N-1]/r[0])/(N-1) */
static void compute_u_coefficients(int N, double mu, double q, double L, double kcrc, double complex *u)
{
  double y = M_PI/L;
  double k0r0 = kcrc * exp(-L);
  double t = -2*y*log(k0r0/2);

  if(q == 0) {
    double x = (mu+1)/2;
    double lnr, phi;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(x, m*y, &lnr, &phi);
      u[m] = polar(1.0,m*t + 2*phi);
    }
  }
  else {
    double xp = (mu+1+q)/2;
    double xm = (mu+1-q)/2;
    double lnrp, phip, lnrm, phim;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(xp, m*y, &lnrp, &phip);
      lngamma_4(xm,-m*y, &lnrm, &phim);
      u[m] = polar(exp(q*M_LN2 + lnrp - lnrm), m*t + phip - phim);
    }
  }

  for(int m = N/2+1; m < N; m++)
    u[m] = conj(u[N-m]);
  if((N % 2) == 0)
    u[N/2] = (creal(u[N/2]) + I*0.0);
}

/* Pre-compute a low-ringing value of k0r0 that appears in the FFTLog implementation of
 * the discrete Hankel transform.  The parameters N, mu, q, spherical_bessel, deriv, and plaw here are the
 * same as for the function general_fht().  The parameter L is defined (for whatever
 * reason) to be N times the logarithmic spacing of the input array, i.e.
 *   L = N * log(r[N-1]/r[0])/(N-1) */
static double goodkr_new_deriv(int N, double mu, double q, double L, int spherical_bessel, double deriv, double plaw, double kr)
{
  double complex prefac;
  q+=plaw;
  double xp = (mu+1+q-deriv-0.5*spherical_bessel)/2;
  double xm = (mu+1-q+deriv+0.5*spherical_bessel)/2;
  double y = M_PI*N/(2*L);
  double lnrp, lnrm, argm, argp;
  lngamma_4(xp, y, &lnrp, &argp);
  lngamma_4(xm, -y, &lnrm, &argm);
  double arg;

  prefac = cpow(2.0, q + 1.0*spherical_bessel+2*I*y-deriv) * cpow(1.0, -2*I*y);//polar(exp(prefac), m*y);

  prefac = (polar(exp( lnrp - lnrm), argp - argm)) * prefac;

  if(deriv>0.0) {
    for (int j = 1; j<=(int)deriv; j++){
        prefac*=(q +2*I*y - (j-spherical_bessel));
      }
  }
  prefac*=pow(-1.0, deriv);


  arg = cimag(clog((prefac)))/M_PI;
  double iarg = round(arg);
  if(arg != iarg)
    kr *= exp((arg - iarg)*L/N);

  return kr;

}

/* Pre-compute the coefficients that appear in the FFTLog implementation of
 * the discrete Hankel transform.  The parameters N, mu, q, spherical_bessel, deriv, and plaw here are the
 * same as for the function general_fht().  The parameter L is defined (for whatever
 * reason) to be N times the logarithmic spacing of the input array, i.e.
 *   L = N * log(r[N-1]/r[0])/(N-1) */
static void compute_u_coefficients_new_deriv(int N, double mu, double q, double L, double kcrc, int spherical_bessel, double deriv, double plaw, double complex *u)
{
  double y = M_PI/L;
  double k0r0 = kcrc * exp(-L);
  double t = -2*y*log(k0r0/2);
  q+=plaw;

  double xp = (mu+1+q-deriv-0.5*spherical_bessel)/2;
  double xm = (mu+1-q+deriv+0.5*spherical_bessel)/2;
  double lnrp, phip, lnrm, phim;
  for(int m = 0; m <= N/2; m++) {
    lngamma_4(xp, m*y, &lnrp, &phip);
    lngamma_4(xm,-m*y, &lnrm, &phim);
    u[m] = cpow(2.0, q + 1.0*spherical_bessel+2*I*m*y - deriv) * cpow(k0r0, -2*I*m*y);
    u[m] *= polar(exp(lnrp - lnrm), phip - phim);
    
    for (int j = 1; j<=(int)deriv; j++){
      u[m]*=(q +2*I*m*y - (j-spherical_bessel));//(q +2*I*m*y - j-1.5);
    }

    u[m]*=pow(-1.0,deriv);
  }
  
  for(int m = N/2+1; m < N; m++)
    u[m] = conj(u[N-m]);
  if((N % 2) == 0)
    u[N/2] = (creal(u[N/2]) + I*0.0);
}


/* Compute the discrete Hankel transform of the function a(r).  See the FFTLog
 * documentation (or the Fortran routine of the same name in the FFTLog
 * sources) for a description of exactly what this function computes.
 * If u is NULL, the transform coefficients will be computed anew and discarded
 * afterwards.  If you plan on performing many consecutive transforms, it is
 * more efficient to pre-compute the u coefficients. */
static void fht(int npk, int N,
    double *k, double **pk,
    double *r, double **xi,
    double dim, double mu, double q, double kcrc,
    int noring, double complex* u, int *status)
{
  fftw_plan forward_plan, reverse_plan;
  double L = log(k[N-1]/k[0]) * N/(N-1.);
  double complex* ulocal = NULL;
  if(u == NULL) {
    if(noring)
      kcrc = goodkr(N, mu, q, L, kcrc);

    ulocal = malloc (sizeof(complex double)*N);
    if(ulocal==NULL)
      *status=CCL_ERROR_MEMORY;

    if(*status == 0) {
      compute_u_coefficients(N, mu, q, L, kcrc, ulocal);
      u = ulocal;
    }
  }

  fftw_complex* a_tmp;
  fftw_complex* b_tmp;
  if(*status == 0) {
    a_tmp = fftw_alloc_complex(N);
    if(a_tmp==NULL)
      *status=CCL_ERROR_MEMORY;
  }
  if(*status == 0) {
    b_tmp = fftw_alloc_complex(N);
    if(b_tmp==NULL)
      *status=CCL_ERROR_MEMORY;
  }

  if(*status == 0) {
    /* Compute the convolution b = a*u using FFTs */
    forward_plan = fftw_plan_dft_1d(N,
                                    (fftw_complex*) a_tmp,
                                    (fftw_complex*) b_tmp,
                                    -1, FFTW_ESTIMATE);
    reverse_plan = fftw_plan_dft_1d(N,
                                    (fftw_complex*) b_tmp,
                                    (fftw_complex*) b_tmp,
                                    +1, FFTW_ESTIMATE);
  }

  if(*status == 0) {
    #pragma omp parallel default(none) \
                         shared(npk, N, k, pk, r, xi, \
                                dim, mu, q, kcrc, u, status, \
                                forward_plan, reverse_plan, \
                                L, ulocal)
    {
      int local_status = 0;

      double *prefac_pk=NULL;
      if(local_status == 0) {
        prefac_pk = malloc(N*sizeof(double));
        if(prefac_pk==NULL)
          local_status=CCL_ERROR_MEMORY;
      }

      double *prefac_xi=NULL;
      if(local_status == 0) {
        prefac_xi = malloc(N*sizeof(double));
        if(prefac_xi==NULL)
          local_status=CCL_ERROR_MEMORY;
      }

      fftw_complex* a=NULL;
      fftw_complex* b=NULL;
      if(local_status == 0) {
        a = fftw_alloc_complex(N);
        if(a==NULL)
        local_status=CCL_ERROR_MEMORY;
      }

      if(local_status == 0) {
        b = fftw_alloc_complex(N);
        if(b==NULL)
          local_status=CCL_ERROR_MEMORY;
      }

      if(local_status == 0) {
        for(int i = 0; i < N; i++)
          prefac_pk[i] = pow(k[i], dim/2-q);

        /* Compute k's corresponding to input r's */
        double k0r0 = kcrc * exp(-L);
        r[0] = k0r0/k[0];
        for(int n = 1; n < N; n++)
          r[n] = r[0] * exp(n*L/N);

        double one_over_2pi_dhalf = pow(2*M_PI,-dim/2);
        for(int i = 0; i < N; i++)
          prefac_xi[i] = one_over_2pi_dhalf * pow(r[i], -dim/2-q);

        #pragma omp for
        for(int j = 0; j < npk; j++) {
          for(int i = 0; i < N; i++)
            a[i] = prefac_pk[i] * pk[j][i];

          fftw_execute_dft(forward_plan,a,b);
          for(int m = 0; m < N; m++)
            b[m] *= u[m] / (double)(N);       // divide by N since FFTW doesn't normalize the inverse FFT
          fftw_execute_dft(reverse_plan,b,b);

          /* Reverse b array */
          double complex tmp;
          for(int n = 0; n < N/2; n++) {
            tmp = b[n];
            b[n] = b[N-n-1];
            b[N-n-1] = tmp;
          }

          for(int i = 0; i < N; i++)
            xi[j][i] = prefac_xi[i] * creal(b[i]);
        }
      }

      free(prefac_pk);
      free(prefac_xi);
      fftw_free(a);
      fftw_free(b);

      if (local_status) {
        #pragma omp atomic write
        *status = local_status;
      }
    } //end omp parallel
  }

  if(*status == 0) {
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(reverse_plan);
  }

  free(ulocal);
  //TODO: free this up
  fftw_free(a_tmp);
  fftw_free(b_tmp);
}

/* Compute the discrete Hankel transform of the function a(r) 
* weighted by a power law and the nth derivative of the 
* (spherical) bessel function. Explicitly, this function computes 
* \tilde{a}(k)= \int \frac{dr}{(xk)^plaw} a(r) (J/j)^(n)_\mu(xk)
* The computed transform will be centered about the peak of the given \mu.
* double bessel_deriv: the nth derivative of the (spherical) bessel function
* int spherical_bessel: 1 indicates spherical bessel functions, 0 bessel functions
* double q: the biasing index
* double plaw: power-law index of (xk) weighting in integral
* NOTE: we ignore factors of (2*pi) found in typical fht algorithm,
*   these factors should be calculated and applied a-posteriori if necessary
*/
static void general_fht(int npk, int N,
    double *k, double **pk,
    double *r, double **xi,
    double mu, double q, double kcrc,
    int spherical_bessel, double bessel_deriv, double plaw, double complex* u, int *status)
{
  q = q-1.0*spherical_bessel;
  kcrc = mu+1.0;
  mu = mu+0.5*spherical_bessel;

  fftw_plan forward_plan, reverse_plan;
  double L = log(k[N-1]/k[0]) * N/(N-1.);
  double complex* ulocal = NULL;
  if(u == NULL) {
    
      kcrc = goodkr_new_deriv(N, mu, q, L, spherical_bessel, bessel_deriv,plaw, kcrc);

    ulocal = malloc (sizeof(complex double)*N);
    if(ulocal==NULL)
      *status=CCL_ERROR_MEMORY;

    if(*status == 0) {
      compute_u_coefficients_new_deriv(N, mu, q, L, kcrc, spherical_bessel, bessel_deriv, plaw,ulocal);
      u = ulocal;
    }
  }

  fftw_complex* a_tmp;
  fftw_complex* b_tmp;
  if(*status == 0) {
    a_tmp = fftw_alloc_complex(N);
    if(a_tmp==NULL)
      *status=CCL_ERROR_MEMORY;
  }
  if(*status == 0) {
    b_tmp = fftw_alloc_complex(N);
    if(b_tmp==NULL)
      *status=CCL_ERROR_MEMORY;
  }

  if(*status == 0) {
    /* Compute the convolution b = a*u using FFTs */
    forward_plan = fftw_plan_dft_1d(N,
                                    (fftw_complex*) a_tmp,
                                    (fftw_complex*) b_tmp,
                                    -1, FFTW_ESTIMATE);
    reverse_plan = fftw_plan_dft_1d(N,
                                    (fftw_complex*) b_tmp,
                                    (fftw_complex*) b_tmp,
                                    +1, FFTW_ESTIMATE);
  }

  if(*status == 0) {
    #pragma omp parallel default(none) \
                         shared(npk, N, k, pk, r, xi, \
                                spherical_bessel, bessel_deriv, mu, q, kcrc, u, status, plaw, \
                                forward_plan, reverse_plan, \
                                L, ulocal)
    {
      int local_status = 0;

      double *prefac_pk=NULL;
      if(local_status == 0) {
        prefac_pk = malloc(N*sizeof(double));
        if(prefac_pk==NULL)
          local_status=CCL_ERROR_MEMORY;
      }

      double *prefac_xi=NULL;
      if(local_status == 0) {
        prefac_xi = malloc(N*sizeof(double));
        if(prefac_xi==NULL)
          local_status=CCL_ERROR_MEMORY;
      }

      fftw_complex* a=NULL;
      fftw_complex* b=NULL;
      if(local_status == 0) {
        a = fftw_alloc_complex(N);
        if(a==NULL)
        local_status=CCL_ERROR_MEMORY;
      }

      if(local_status == 0) {
        b = fftw_alloc_complex(N);
        if(b==NULL)
          local_status=CCL_ERROR_MEMORY;
      }

      if(local_status == 0) {
        for(int i = 0; i < N; i++)
          prefac_pk[i] = pow(k[i], -1*spherical_bessel-q);

        /* Compute k's corresponding to input r's */
        double k0r0 = kcrc * exp(-L);
        r[0] = k0r0/k[0];
        for(int n = 0; n < N; n++)
          r[n] = (kcrc)/k[N-1-n];//r[0]* exp(n*L/N);

        //double one_over_2pi_dhalf = pow(2*M_PI,-dim/2);
        for(int i = 0; i < N; i++)
          prefac_xi[i] =  pow(r[i], -1*spherical_bessel-q);

        #pragma omp for
        for(int j = 0; j < npk; j++) {
          for(int i = 0; i < N; i++)
            a[i] = prefac_pk[i] * pk[j][i];

          fftw_execute_dft(forward_plan,a,b);
          for(int m = 0; m < N; m++){

            b[m] *= u[m] / (double)(N);       // divide by N since FFTW doesn't normalize the inverse FFT

          }
          fftw_execute_dft(reverse_plan,b,b);

          /* Reverse b array */
          double complex tmp;
          for(int n = 0; n < N/2; n++) {
            tmp = b[n];
            b[n] = b[N-n-1];
            b[N-n-1] = tmp;
          }

          for(int i = 0; i < N; i++)
            xi[j][i] = prefac_xi[i] * creal(b[i])*pow(pow(M_PI,0.5)/4, spherical_bessel);
        }
      }

      free(prefac_pk);
      free(prefac_xi);
      fftw_free(a);
      fftw_free(b);

      if (local_status) {
        #pragma omp atomic write
        *status = local_status;
      }
    } //end omp parallel
  }

  if(*status == 0) {
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(reverse_plan);
  }

  free(ulocal);
  //TODO: free this up
  fftw_free(a_tmp);
  fftw_free(b_tmp);
}


void ccl_fftlog_ComputeXi2D(double mu, double epsilon,
          int npk, int N, double *l,double **cl,
          double *th, double **xi, int *status)
{
  fht(npk, N, l, cl, th, xi, 2., mu, epsilon, 1, 1, NULL, status);
}

void ccl_fftlog_ComputeXi3D(double l, double epsilon,
          int npk, int N, double *k, double **pk,
          double *r, double **xi, int *status)
{
  fht(npk, N, k, pk, r, xi, 3., l+0.5, epsilon, 1, 1, NULL, status);
}

void ccl_fftlog_ComputeXi_general(double mu, double q, 
  int npk, int N, double *k, double **pk,
  int spherical_bessel, double bessel_deriv, double plaw, 
    double *r, double **xi, int *status)
{
  general_fht(npk, N, k, pk, r, xi, mu, q, 1, spherical_bessel, bessel_deriv, plaw, NULL, status);
}
