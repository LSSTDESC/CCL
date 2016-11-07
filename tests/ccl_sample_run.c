#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define OC 0.25
#define OB 0.05
#define OL 0.70
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define AS 2.1E-9
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 500

int main(int argc,char **argv)
{
  //Initialize cosmological parameters
  printf("Initializing cosmological parameters with call for <ccl_parameters_create>\n\n");
  printf("\tccl_parameters params=ccl_parameters_create(0.25,0.05,0.00,0.00,-1.00,0.00,0.70,2.1E-9,0.96,-1,NULL,NULL);\n\n");
  system("read");
  ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);

  return 0;
}
