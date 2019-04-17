/** @file */
#ifndef __CCL_EH98_H_INCLUDED__
#define __CCL_EH98_H_INCLUDED__

CCL_BEGIN_DECLS

typedef struct eh_struct {
  double rsound;
  double zeq;
  double keq;
  double zdrag;
  double kSilk;
  double rsound_approx;
  double th2p7;
  double alphac;
  double alphab;
  double betac;
  double betab;
  double bnode;
  int wiggled;
} eh_struct;

/*
 * Allocate a new struct for storing EH98 data
 * @param params Cosmological parameters
 * @param int, include BAO wiggles if not 0, smooth otherwise
 */
eh_struct* ccl_eh_struct_new(ccl_parameters *params, int wiggled);

/*
 * Compute the Eisenstein & Hu (1998) unnormalize power spectrum
 * @param params Cosmological parameters
 * @param eh, an eh_struct instance
 * @param k, wavenumber in Mpc^-1
 */
double ccl_eh_power(ccl_parameters *params, eh_struct* eh, double k);

CCL_END_DECLS

#endif
