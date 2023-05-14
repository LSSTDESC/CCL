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

*****************************************************************/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.69314718056
#endif
/* This code is FFTLog, which is described in arXiv:astro-ph/9905191 */

static double complex complex_mult(double complex a, double complex b){
  double a_real = creal(a);
  double b_real = creal(b);
  double a_imag = cimag(a);
  double b_imag = cimag(b);

  return (a_real*b_real - a_imag*b_imag) + (a_real*b_imag + b_real*a_imag)*I;


}
void window_cfft(double complex *out, double c_window_width, int N, double *k) {
  // 'out' is (halfN+1) complex array
  int Ncut;
  //N=1500;
  //N*=2;
  Ncut = (int)(N * 0.5);
  double k_low = k[Ncut];
  double k_high = k[N-Ncut-1];
  int i;
  double W;
  double diff;  
  for(i=0; i<=Ncut; i++) { // window for right-side
    //diff = (k[N-1] - k[N-1-i])/(k[N-1] - k_high);
    //W = diff - 1./(2.*M_PI) * sin(2.*diff*M_PI);
     W = (double)(i)/Ncut - 1./(2.*M_PI) * sin(2.*i*M_PI/Ncut);
    out[N-i] *=W;// W*creal(out[i]) + I*cimag(out[i]);
    printf("%d %f %f %f %f\n", i,k[i], W, creal(out[N-i]), cimag(out[N-i]));

  }
}

double complex lngamma_lanczos_cfft(double complex z) {
/* Lanczos coefficients for g = 7 */
  static double p[] = {
    0.99999999999980993227684700473478,
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7};

  if(creal(z) < 0.5) {return clog(M_PI) - clog(csin(M_PI*z)) - lngamma_lanczos_cfft(1. - z);}
  z -= 1;
  double complex x = p[0];
  for(int n = 1; n < 9; n++){ x += p[n] / (z + (double)(n));}

  double complex t = z + 7.5;
  return log(2*M_PI) /2.  + (z+0.5)*clog(t) -t + clog(x);
}


static double complex ln_g_m_vals_cfft(double mu, double complex q) {
/* similar routine as python version.
use asymptotic expansion for large |mu+q| */
  double complex asym_plus = (mu+1+ q)/2.;
  double complex asym_minus= (mu+1- q)/2.;
  return (asym_plus-0.5)*clog(asym_plus) - (asym_minus-0.5)*clog(asym_minus) - q \
    +1./12 *(1./asym_plus - 1./asym_minus) \
    +1./360.*(1./cpow(asym_minus,3) - 1./cpow(asym_plus,3)) \
    +1./1260*(1./cpow(asym_plus,5) - 1./cpow(asym_minus,5));  
}

void g_l_1_cfft(double l, double nu, double *eta, double complex *gl1, long N) {
/* z = nu + I*eta
Calculate g_l_1 = exp(zln2 + lngamma( (l+nu-1)/2 + I*eta/2 ) - lngamma( (4+l-nu)/2 - I*eta/2 ) ) */
  long i;
  double complex z;
  for(i=0; i<N; i++) {
    z = nu+I*eta[i];
    if(l+fabs(eta[i])<200){
      gl1[i] = -(z-1.)* cexp((z-1.)*log(2.) + lngamma_lanczos_cfft((l+z-1.)/2.) - lngamma_lanczos_cfft((4.+l-z)/2.));
    }else{
      gl1[i] = -(z-1.)* cexp((z-1.)*log(2.) + ln_g_m_vals_cfft(l+0.5, z-2.5));
    }
  }
}
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


static void lngamma_4_approx(double x, double y, double* lnr, double* arg)
{
  double complex w;
  w = ln_g_m_vals_cfft(x+0.5, (y-1.5));
  
  if(lnr) *lnr = creal(w);
  if(arg) *arg = cimag(w);
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
static double goodkr_new(int N, double mu, double q, double L, double deriv, double kr)
{
  double complex prefac;

  if (deriv==0){
  double xp = (mu+1+q)/2;
  double xm = (mu+1-q)/2;
  double y = M_PI*N/(2*L);
  double lnrp, lnrm, argm, argp;
  lngamma_4(xp, y, &lnrp, &argp);
  lngamma_4(xm, -y, &lnrm, &argm);
  double arg = log(2/kr) * N/L + (argp - argm)/M_PI;

  prefac = cpow(2.0, q + 1.5+2*I*y-deriv) * cpow(1.0, -2*I*y);//polar(exp(prefac), m*y);

  prefac = complex_mult(polar(exp( lnrp - lnrm), argp - argm), prefac);
  arg = cimag(clog((prefac)))/M_PI;
  double iarg = round(arg);
  if(arg != iarg)
    kr *= exp((arg - iarg)*L/N);
  }
  else{
  double xp_up = (mu+1+q+deriv)/2;
  double xm_up = (mu+1-q+deriv)/2;
  double xp_down = (mu+1+q-deriv)/2;
  double xm_down = (mu+1-q-deriv)/2;
  double y = M_PI*N/(2*L);
  double lnr, argm_up, argp_up, argm_down, argp_down;
  lngamma_4(xp_up, y, &lnr, &argp_up);
  lngamma_4(xm_up, y, &lnr, &argm_up);
  lngamma_4(xp_down, y, &lnr, &argp_down);
  lngamma_4(xm_down, y, &lnr, &argm_down);
  double arg = log(2/kr) * N/L + (argp_up + argm_up)/M_PI + (argp_down + argm_down)/M_PI;
  double iarg = round(arg);
  if(arg != iarg)
    kr *= exp((arg - iarg)*L/N);   


  }
  return kr;

}
static double goodkr_spherical(int N, double mu, double q, double L, double deriv, double kr)
{
  //q = q-1;
  double limit = (mu+q);
  double xp = (mu+q-deriv)/2;
  double xm = (mu+3-q+deriv)/2;
  double y = M_PI*N/(2*L);
  double lnrp, lnrm, argm, argp;
  kr = 1;
  double complex prefac;
  double pre_lnrp, pre_phip, pre_lnrm, pre_phim;

  lngamma_4(xp, y, &lnrp, &argp);
  lngamma_4(xm, -y, &lnrm, &argm);
  double arg = log(2/kr) * N/L + (argp - argm)/M_PI;
  prefac = cpow(2.0, q + 2*I*y-deriv);//polar(exp(prefac), m*y);

  prefac = complex_mult(polar(exp( lnrp - lnrm), argp - argm), prefac);

  //printf("%f\n", u[m]);
  if (deriv>0.){
    lngamma_4(q, y, &pre_lnrp, &pre_phip);
    lngamma_4(q-deriv, y, &pre_lnrm, &pre_phim);
    limit = polar(exp(pre_lnrp - pre_lnrm), (pre_phip - pre_phim));
    prefac = complex_mult(limit, prefac);
    prefac *= pow(-1., deriv);

  }
  arg = cimag(clog((prefac)))/M_PI;

  double iarg = round(arg);  
  if(arg != iarg)
    kr *= exp((arg - iarg)*L/N);
  return kr;
}


/* Pre-compute the coefficients that appear in the FFTLog implementation of
 * the discrete Hankel transform for a *spherical* bessel function.  The parameters N, mu, and q here are the
 * same as for the function fht().  The parameter L is defined (for whatever
 * reason) to be N times the logarithmic spacing of the input array, i.e.
 *   L = N * log(r[N-1]/r[0])/(N-1) 
 * Computes the function \int dx x^{q-1} \ell j_\mu(x) = 2^q \Gamma[(\mu + q) / 2] / \Gamma[(\mu + 3 - q) / 2] /
 */
static void compute_u_coefficients_spherical(int N, double mu, double q, double L, double kcrc, double deriv, double complex *u)
{
    double complex limit;
  double y = M_PI/L;
  double k0r0 = kcrc * exp(-L);
  double t = -2*y*log(k0r0/2);
  if((q>2)||(mu<deriv &&q<-mu)||(mu>=deriv &&q<(deriv-mu))){
    printf("power-law in FFTLog not valid for spherical bessel function integral/derivative. Exiting. \n");
    exit(1);
  }

  double prefac;
  prefac = (q)*M_LN2;
  double pre_lnrp, pre_phip, pre_lnrm, pre_phim;
  double xp = (mu+q-deriv)/2;
  double xm = (mu+3-q+deriv)/2;
  double lnrp, phip, lnrm, phim;
  //double* eta = malloc (sizeof(double)*N);
  for(int m = 0; m <= N/2; m++) {
    u[m] = cpow(2.0, q + 2*I*m*y-(deriv));//polar(exp(prefac), m*y);
    //eta[m] = m*y;
    lngamma_4(xp, m*y, &lnrp, &phip);
    lngamma_4(xm,-m*y, &lnrm, &phim);
    u[m] =complex_mult(polar(exp(lnrp - lnrm), phip - phim), u[m]);
      //if(deriv==0.0){
      //printf("%f + i%f, %d, %f\n",creal(u[m]), cimag(u[m]), m,m*y);}
    //printf("%f\n", u[m]);
    if (deriv>0.){
      lngamma_4(q, 2*m*y, &pre_lnrp, &pre_phip);
      lngamma_4(q-deriv, 2*m*y, &pre_lnrm, &pre_phim);
      limit = polar(exp(pre_lnrp - pre_lnrm), (pre_phip - pre_phim));
      u[m] = complex_mult(limit, u[m]);
      //u[m] = complex_mult(u[m], polar(1.0, -m*y*M_LN2));
      //u[m] *= pow(-1., deriv)*pow(0.5, deriv);
      u[m] *=pow(-1., deriv);// * pow(0.5,deriv) + 0.0*I);


      //printf("%f %f %f %f\n", pre_lnrp, pre_lnrm, pre_phip, pre_phim);}
    }


    }
  
  for(int m = N/2+1; m < N; m++)
    u[m] = conj(u[N-m]);
  if((N % 2) == 0)
    u[N/2] = (creal(u[N/2]) + I*0.0);
  //g_l_1_cfft(mu, q, eta, u, (long) N);
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


static void compute_u_coefficients_new(int N, double mu, double q, double L, double kcrc, double deriv, double complex *u)
{
  double y = M_PI/L;
  double k0r0 = kcrc * exp(-L);
  double t = -2*y*log(k0r0/2);

  if(q == 0) {
    double x = (mu+1+deriv)/2;
    double lnr, phi;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(x, m*y, &lnr, &phi);
      u[m] = polar(1.0,m*t + 2*phi);
    }
  }

  if(deriv==0.0){
    double xp = (mu+1+q)/2;
    double xm = (mu+1-q)/2;
    double lnrp, phip, lnrm, phim;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(xp, m*y, &lnrp, &phip);
      lngamma_4(xm,-m*y, &lnrm, &phim);
      u[m] = cpow(2.0, q + 1.5+2*I*m*y) * cpow(k0r0, -2*I*m*y);
      u[m] *= polar(exp(lnrp - lnrm), phip - phim);
    }
  }
  else {
    double xp_up = (mu+1+q+deriv)/2;
    double xm_up = (mu+1-q+deriv)/2;
    double xp_down = (mu+1+q-deriv)/2;
    double xm_down = (mu+1-q-deriv)/2;
    //double y = M_PI*N/(2*L);
    double argm_up, argp_up, argm_down, argp_down;
    double lnrp_up,lnrm_up,lnrp_down,lnrm_down;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(xp_up, m*y, &lnrp_up, &argp_up);
      lngamma_4(xm_up, -m*y, &lnrm_up, &argm_up);
      lngamma_4(xp_down, m*y, &lnrp_down, &argp_down);
      lngamma_4(xm_down, -m*y, &lnrm_down, &argm_down);
      u[m] = 0.5*(polar(exp(q*M_LN2 + lnrp_up - lnrm_up), m*t + argp_up - argm_up));
      u[m] = u[m] - 0.5*(polar(exp(q*M_LN2 + lnrp_down - lnrm_down), m*t + argp_down - argm_down));
    }  
  }
  for(int m = N/2+1; m < N; m++)
    u[m] = conj(u[N-m]);
  if((N % 2) == 0)
    u[N/2] = (creal(u[N/2]) + I*0.0);
}

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
  double arg = log(2/kr) * N/L + (argp - argm)/M_PI;

  prefac = cpow(2.0, q + 1.0*spherical_bessel+2*I*y-deriv) * cpow(1.0, -2*I*y);//polar(exp(prefac), m*y);

  prefac = complex_mult(polar(exp( lnrp - lnrm), argp - argm), prefac);

  if(deriv>0.0) {
      //for (int j = 1; j<=(int)deriv; j++){
      //  prefac*=(q-plaw +2*I*y - j-1.5);
      //}

    for (int j = 1; j<=(int)deriv; j++){
        prefac*=(q-plaw +2*I*y - (j-spherical_bessel));
      }

  }
  //if (plaw==-2.0){
  //  prefac/=((mu-2.+q +2*I*y)*(3.+mu-q -2*I*y));
  //}
  prefac*=pow(-1.0, deriv);


  arg = cimag(clog((prefac)))/M_PI;
  double iarg = round(arg);
  if(arg != iarg)
    kr *= exp((arg - iarg)*L/N);

  return kr;

}
static void compute_u_coefficients_new_deriv(int N, double mu, double q, double L, double kcrc, int spherical_bessel, double deriv, double plaw, double complex *u)
{
  double y = M_PI/L;
  double k0r0 = kcrc * exp(-L);
  double t = -2*y*log(k0r0/2);
  q+=plaw;

  if(q == 0) {
    double x = (mu+1+deriv)/2;
    double lnr, phi;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(x, m*y, &lnr, &phi);
      u[m] = polar(1.0,m*t + 2*phi);
    }
  }


    double xp = (mu+1+q-deriv-0.5*spherical_bessel)/2;
    double xm = (mu+1-q+deriv+0.5*spherical_bessel)/2;
    double lnrp, phip, lnrm, phim;
    for(int m = 0; m <= N/2; m++) {
      lngamma_4(xp, m*y, &lnrp, &phip);
      lngamma_4(xm,-m*y, &lnrm, &phim);
      u[m] = cpow(2.0, q + 1.0*spherical_bessel+2*I*m*y - deriv) * cpow(k0r0, -2*I*m*y);
      u[m] *= polar(exp(lnrp - lnrm), phip - phim);
      //for (int j = 1; j<=(int)deriv; j++){
      //  u[m]*=(q-plaw +2*I*m*y - j-1.5);
      //}

      for (int j = 1; j<=(int)deriv; j++){
        u[m]*=(q-plaw +2*I*m*y - (j-spherical_bessel));//(q +2*I*m*y - j-1.5);
      }
      u[m]*=pow(-1.0,deriv);
      //if (plaw==-2.0){
      //  u[m]/=((mu-2.+q +2*I*m*y)*(3.+mu-q -2*I*m*y));
      //}

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
      kcrc = goodkr_new(N, mu, q, L, 0.0,kcrc);

    ulocal = malloc (sizeof(complex double)*N);
    if(ulocal==NULL)
      *status=CCL_ERROR_MEMORY;

    if(*status == 0) {
      compute_u_coefficients_new(N, mu, q, L, kcrc, 0.0, ulocal);
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
          prefac_pk[i] = pow(k[i], -1.5-q);

        /* Compute k's corresponding to input r's */
        double k0r0 = kcrc * exp(-L);
        r[0] = k0r0/k[0];
        for(int n = 0; n < N; n++)
          r[n] = (kcrc)/k[N-1-n];//r[0]* exp(n*L/N);

        double one_over_2pi_dhalf = pow(2*M_PI,-dim/2);
        for(int i = 0; i < N; i++)
          prefac_xi[i] =  pow(r[i], -1.5-q);

        #pragma omp for
        for(int j = 0; j < npk; j++) {
          for(int i = 0; i < N; i++)
            a[i] = prefac_pk[i] * pk[j][i];

          fftw_execute_dft(forward_plan,a,b);
          //window_cfft(b, 0.25, N/2, k);
          for(int m = 0; m < N; m++){
          //printf("i: %d, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", m, m*2*M_PI/L, creal(b[m]), cimag(b[m]),creal(u[m]), cimag(u[m]));

            b[m] *= u[m] / (double)(N);       // divide by N since FFTW doesn't normalize the inverse FFT
          //printf("i: %d, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", m, m*M_PI/L, creal(b[m]), cimag(b[m]),creal(u[m]), cimag(u[m]));

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
            xi[j][i] = prefac_xi[i] * creal(b[i])*pow(M_PI,0.5)/4;
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
/* Compute the discrete Hankel transform of the function a(r).  See the FFTLog
 * documentation (or the Fortran routine of the same name in the FFTLog
 * sources) for a description of exactly what this function computes.
 * If u is NULL, the transform coefficients will be computed anew and discarded
 * afterwards.  If you plan on performing many consecutive transforms, it is
 * more efficient to pre-compute the u coefficients. */
static void fht_first_deriv(int npk, int N,
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
      kcrc = goodkr_new_first_deriv(N, mu, q, L, dim,kcrc);

    ulocal = malloc (sizeof(complex double)*N);
    if(ulocal==NULL)
      *status=CCL_ERROR_MEMORY;

    if(*status == 0) {
      compute_u_coefficients_new_first_deriv(N, mu, q, L, kcrc, dim, ulocal);
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
          prefac_pk[i] = pow(k[i], -1-q);

        /* Compute k's corresponding to input r's */
        double k0r0 = kcrc * exp(-L);
        r[0] = k0r0/k[0];
        for(int n = 0; n < N; n++)
          r[n] = (kcrc)/k[N-1-n];//r[0]* exp(n*L/N);

        double one_over_2pi_dhalf = pow(2*M_PI,-dim/2);
        for(int i = 0; i < N; i++)
          prefac_xi[i] =  pow(r[i], -1-q);

        #pragma omp for
        for(int j = 0; j < npk; j++) {
          for(int i = 0; i < N; i++)
            a[i] = prefac_pk[i] * pk[j][i];

          fftw_execute_dft(forward_plan,a,b);
          //window_cfft(b, 0.25, N/2, k);
          for(int m = 0; m < N; m++){
          //printf("i: %d, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", m, m*2*M_PI/L, creal(b[m]), cimag(b[m]),creal(u[m]), cimag(u[m]));

            b[m] *= u[m] / (double)(N);       // divide by N since FFTW doesn't normalize the inverse FFT
          //printf("i: %d, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", m, m*M_PI/L, creal(b[m]), cimag(b[m]),creal(u[m]), cimag(u[m]));

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
            xi[j][i] = prefac_xi[i] * creal(b[i])*pow(M_PI,0.5)/4;
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
/* Compute the discrete Hankel transform of the function pk 
* weighted by a power law and the nth derivative of the 
* (spherical) bessel function. Explicitly, this function computes 
* \tilde{P}(x)= \int \frac{dk}{k^q} P(k) [choice of bessel and its derivative]_\mu(xk)*/
static void general_fht(int npk, int N,
    double *k, double **pk,
    double *r, double **xi,
    double mu, double q, double kcrc,
    int spherical_bessel, double bessel_deriv, double plaw, double complex* u, int *status)
{
  //if (bessel_deriv==0.0) fht(npk, N, k,pk,r,xi,0.0,mu+0.5, q-1.5, mu+1.0, 1, NULL, status);
  //else fht_first_deriv(npk, N, k,pk,r,xi,bessel_deriv,mu+0.5, q-1.0, mu+1.0, 1, NULL, status);
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
          //window_cfft(b, 0.25, N/2, k);
          for(int m = 0; m < N; m++){
          //printf("i: %d, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", m, m*2*M_PI/L, creal(b[m]), cimag(b[m]),creal(u[m]), cimag(u[m]));

            b[m] *= u[m] / (double)(N);       // divide by N since FFTW doesn't normalize the inverse FFT
          //printf("i: %d, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", m, m*M_PI/L, creal(b[m]), cimag(b[m]),creal(u[m]), cimag(u[m]));

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
  int spherical_bessel, double bessel_deriv, double window_frac, 
    double *r, double **xi, int *status)
{

  general_fht(npk, N, k, pk, r, xi, mu, q, 1, spherical_bessel, bessel_deriv, window_frac, NULL, status);
  //printf("npk: %d, N: %d, mu: %f, q: %f, bessel: %d, deriv: %f, frac: %f\n", npk, N, mu, q, spherical_bessel, bessel_deriv, window_frac);
}