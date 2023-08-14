#include <complex.h>
#include <fftw3.h>

void g_l(double l, double nu, double *eta, double complex *gl, long N);
void g_l_1(double l, double nu, double *eta, double complex *gl1, long N);
void g_l_2(double l, double nu, double *eta, double complex *gl2, long N);

void g_l_modified(double l, double nu, double *eta, double complex *gl, long N);
void g_l_1_modified(double l, double nu, double *eta, double complex *gl1, long N);
void g_l_2_modified(double l, double nu, double *eta, double complex *gl2, long N);



void c_window(double complex *out, double c_window_width, long halfN);

// void resample_fourier_gauss(double *k, double *fk, config *config);

double complex gamma_lanczos(double complex z);
double complex lngamma_lanczos(double complex z);
double complex ln_g_m_vals(double mu, double complex q);