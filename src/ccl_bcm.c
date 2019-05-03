#include <math.h>
#include "ccl.h"

/* BCM correction */
// See Schneider & Teyssier (2015) for details of the model.
double ccl_bcm_model_fka(ccl_cosmology * cosmo, double k, double a, int *status) {
  double fkz;
  double b0;
  double bfunc, bfunc4;
  double kg;
  double gf,scomp;
  double kh;
  double z;

  z = 1./a - 1.;
  kh = k / cosmo->params.h;
  b0 = 0.105*cosmo->params.bcm_log10Mc - 1.27;
  bfunc = b0 / (1. + pow(z/2.3, 2.5));
  bfunc4 = (1-bfunc) * (1-bfunc) * (1-bfunc) * (1-bfunc);
  kg = 0.7 * bfunc4 * pow(cosmo->params.bcm_etab, -1.6);
  gf = bfunc / (1 + pow(kh/kg, 3.)) + 1. - bfunc; //k in h/Mpc
  scomp = 1 + (kh / cosmo->params.bcm_ks) * (kh / cosmo->params.bcm_ks); //k in h/Mpc
  fkz = gf * scomp;
  return fkz;
}
