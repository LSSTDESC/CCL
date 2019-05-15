#include "common.h"

int main(int argc,char **argv)
{ 
  char fname_ini[256];
  if(argc!=2) {
    printf("Usage: limberjack param_file\n");
    exit(0);
  }
  sprintf(fname_ini,"%s",argv[1]);

  RunParams *par=init_params(fname_ini);
  compute_spectra(par);
  compute_w_theta(par);
  write_output(par);
  param_free(par);
  return 0;
}
