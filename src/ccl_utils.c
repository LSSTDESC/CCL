#include "ccl_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ccl_params.h"

/* ------- ROUTINE: ccl_linear spacing ------
INPUTS: [xmin,xmax] of the interval to be divided in N bins
OUTPUT: bin edges in range [xmin,xmax]
*/

double * ccl_linear_spacing(double xmin, double xmax, int N){
    double dx = (xmax-xmin)/(N -1.);

    double * x = malloc(sizeof(double)*N);
    if (x==NULL){
        fprintf(stderr, "ERROR: Could not allocate memory for linear-spaced array (N=%d)\n", N);
        return x;
    }

    for (int i=0; i<N; i++){
        x[i] = xmin + dx*i;
    }
    x[0]=xmin; //Make sure roundoff errors don't spoil edges
    x[N-1]=xmax; //Make sure roundoff errors don't spoil edges

    return x;
}

/* ------- ROUTINE: ccl_log spacing ------
INPUTS: [xmin,xmax] of the interval to be divided logarithmically in N bins
TASK: divide an interval in N logarithmic bins
OUTPUT: bin edges in range [xmin,xmax]
*/

double * ccl_log_spacing(double xmin, double xmax, int N){

    if (N<2){
        fprintf(stderr, "ERROR: Cannot make log-spaced array with %d points - need at least 2\n", N);
        return NULL;
    }

    if (!(xmin>0 && xmax>0)){
        fprintf(stderr, "ERROR: Cannot make log-spaced array xmax or xmax non-positive (had %le, %le)\n", xmin, xmax);
        return NULL;
    }


    double log_xmax = log(xmax);
    double log_xmin = log(xmin);
    double dlog_x = (log_xmax - log_xmin) /  (N-1.);


    double * x = malloc(sizeof(double)*N);
    if (x==NULL){
        fprintf(stderr, "ERROR: Could not allocate memory for log-spaced array (N=%d)\n", N);
        return x;
    }


    for (int i=0; i<N; i++){
        x[i] = exp(log_xmin + dlog_x*i);
    }
    x[0]=xmin; //Make sure roundoff errors don't spoil edges
    x[N-1]=xmax; //Make sure roundoff errors don't spoil edges

    return x;
}


