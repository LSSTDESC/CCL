//
//  emu.c
//  
//
//  Created by Earl Lawrence on 11/10/16.
//  Modified by Kelly Moran on 5/2/22.
//
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "params.h"

// Sizes of stuff
static int m=111, neta=2808, peta=45, rs=8, p=8, nmode=351;

// Kriging basis computed by emuInit
// Sizes of each basis will be peta and m
static double KrigBasis[45][111];

// Initialization to compute the kriging basis for the two parts
void emuInit() {
   
    int ee, i, j, k, l;
    double cov;
    gsl_matrix *SigmaSim;
    gsl_vector *b;
    
    // This emu has just one part using all m of the data for the the peta components
        
    // Allocate some stuff
    SigmaSim = gsl_matrix_alloc(m, m);
    b = gsl_vector_alloc(m);
    
    // Loop over the basis
    for(i=0; i<peta; i++) {
            
        // Loop over the number of simulations
        for(j=0; j<m; j++) {
            
            // Diagonal term
            gsl_matrix_set(SigmaSim, j, j, (1.0/lamz[i]) + (1.0/lamws[i]));
            
            // Off-diagonals
            for(k=0; k<j; k++) {
                
                // compute the covariance
                cov = 0.0;
                for(l=0; l<p; l++) {
                    cov -= beta[i][l]*pow(x[j][l] - x[k][l], 2.0);
                } // for(l=0; l<p; l++)
                cov = exp(cov) / lamz[i];
                
                // put the covariance where it belongs
                gsl_matrix_set(SigmaSim, j, k, cov);
                gsl_matrix_set(SigmaSim, k, j, cov);
                
            } // for(k=0; k<j; k++)
            
            // Vector for the PC weights
            gsl_vector_set(b, j, w[i][j]);
            
        } // for(j=0; j<m; j++)
        
        // Cholesky and solve
        gsl_linalg_cholesky_decomp(SigmaSim);
        gsl_linalg_cholesky_svx(SigmaSim, b);
        
        // Put b where it belongs in the Kriging basis
        for(j=0; j<m; j++) {
            KrigBasis[i][j] = gsl_vector_get(b, j);
        }
        
    } // for(i=0; i<peta; i++)
    
    // Clean this up
    gsl_matrix_free(SigmaSim);
    gsl_vector_free(b);
    
} // emuInit()

// Actual emulation
void emu(double *xstar, double *ystar) {
    
    static double inited=0;
    int ee, i, j, k;
    double wstar[peta];
    double Sigmastar[peta][m];
    double ystaremu[neta];
    double ybyz[rs];
    double logc;
    double xstarstd[p];
    int zmatch;
    gsl_spline *zinterp = gsl_spline_alloc(gsl_interp_cspline, rs);
    gsl_interp_accel *accel = gsl_interp_accel_alloc();
    
    
    // Initialize if necessary
    if(inited==0) {
        emuInit();
        inited = 1;
    }
    
    // Transform w_a into (-w_0-w_a)^(1/4)
    xstar[6] = pow(-xstar[5]-xstar[6], 0.25);
    
    // Check the inputs to make sure we're interpolating.
    for(i=0; i<p; i++) {
        if((xstar[i] < xmin[i]) || (xstar[i] > xmax[i])) {
            switch(i) {
                case 0:
                    printf("omega_m must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 1:
                    printf("omega_b must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 2:
                    printf("sigma_8 must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 3:
                    printf("h must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 4:
                    printf("n_s must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 5:
                    printf("w_0 must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 6:
                    printf("(-w_0-w_a)^(1/4) must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
                case 7:
                    printf("omega_nu must be between %f and %f.\n", xmin[i], xmax[i]);
                    break;
            }
            exit(1);
        }
    } // for(i=0; i<p; i++)
    // Note z in params.h should be in DESCENDING order
    if((xstar[p] < z[rs-1]) || (xstar[p] > z[0])) {
        printf("z must be between %f and %f.\n", z[rs-1], z[0]);
        printf("user input z: %f\n", xstar[p]);
        exit(1);
    }
    
    // Standardize the inputs
    for(i=0; i<p; i++) {
        xstarstd[i] = (xstar[i] - xmin[i]) / xrange[i];
    }
    
    // Compute the covariances between the new input and sims for all the PCs.
    for(i=0; i<peta; i++) {
        for(j=0; j<m; j++) {
            logc = 0.0;
            for(k=0; k<p; k++) {
                logc -= beta[i][k]*pow(x[j][k] - xstarstd[k], 2.0);
            }
            Sigmastar[i][j] = exp(logc) / lamz[i];
        }
    }
    
    // Compute wstar
    for(i=0; i<peta; i++) {
        wstar[i]=0.0;
        for(j=0; j<m; j++) {
            wstar[i] += Sigmastar[i][j] * KrigBasis[i][j];
        }
    }
    
    // Compute ystar, the new output
    for(i=0; i<neta; i++) {
        ystaremu[i] = 0.0;
        for(j=0; j<peta; j++) {
            ystaremu[i] += K[i][j]*wstar[j];
        }
        ystaremu[i] = ystaremu[i]*sd + mean[i];
    }
    
    // Interpolate to the desired redshift
    // Natural cubic spline interpolation over z.
    
    // First check to see if the requested z is one of the training z.
    zmatch = -1;
    for(i=0; i<rs; i++) {
        if(xstar[p] == z[i]) {
            zmatch = i;
        }
    }
    
    // z doesn't match a training z, interpolate
    if(zmatch == -1) {
        for(i=0; i<nmode; i++) {
            for(j=0; j<rs; j++) {
                ybyz[rs-j-1] = ystaremu[j*nmode+i];
            }
            gsl_spline_init(zinterp, z_asc, ybyz, rs);
            ystar[i] = gsl_spline_eval(zinterp, xstar[p], accel);
            gsl_interp_accel_reset(accel);
        }
        
        gsl_spline_free(zinterp);
        gsl_interp_accel_free(accel);
    } else { //otherwise, copy in the emulated z without interpolating
        for(i=0; i<nmode; i++) {
            ystar[i] = ystaremu[zmatch*nmode + i];
        }
    }
    
    // Convert to P(k)
    for(i=0; i<nmode; i++) {
        ystar[i] = ystar[i] - 1.5*log10(mode[i]) + log10(2) + 2*log10(M_PI);
        ystar[i] = pow(10, ystar[i]);
    }
}

int main(int argc, char **argv) {
    
    // A main function to be used as an example.
    
    // Parameter order
    // '\omega_m'   '\omega_b'   '\sigma_8'   'h'   'n_s'   'w_0'   'w_a'   '\omega_{\nu}'   'z'
    
    double xstar[9]; // = {0.1335, 0.02258, 0.8, 0.71, 0.963, -1.0, 0.0, 0.0, .75};
    double ystar[351];
    int i, j;
    FILE *infile;
    FILE *outfile;
    char instring[256];
    char outname[256];
    char *token;
    int good = 1;
    int ctr = 0;
    char ctrc[100];
    
    // Read inputs from a file
    // File should be space delimited with 9 numbers on each line
    // '\omega_m'   '\omega_b'   '\sigma_8'   'h'   'n_s'   'w_0'   'w_a'   '\omega_{\nu}'   'z'
    if((infile = fopen("xstar.dat","r"))==NULL) {
        printf("Cannot find inputs.\n");
        exit(1);
    }
    
    // Read in the inputs and emulate the results.
    while(good == 1) {
        
        // Read each line
        if(fgets(instring, 256, infile) != NULL) {
            token = strtok(instring, " ");
            
            // Parse each line, which is space delimited
            for(i=0; i<9; i++) {
                xstar[i] = atof(token);
                token = strtok(NULL, " ");
            }
            
            // Get the answer.
            emu(xstar, ystar);
            
            // output file name
            strcpy(outname, "EMU");
            sprintf(ctrc, "%i", ctr);
            strcat(outname, ctrc);
            strcat(outname, ".txt");
            
            // Open the output file
            if ((outfile = fopen(outname,"w"))==NULL) {
                printf("cannot open %s \n",outname);
                exit(1);
            }
            for(i=0; i<nmode; i++) {
                fprintf(outfile, "%f %f \n", mode[i], ystar[i]);
            }
            fclose(outfile);
            
            ctr++;
        } else {
            good = 0;
        }
    }
    fclose(infile);
    
}
