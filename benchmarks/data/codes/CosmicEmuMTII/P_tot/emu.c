//
//  emu.c
//  
//
//  Created by Earl Lawrence on 11/10/16.
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
static int m[2] = {111, 36}, neta=2808, peta[2]={7, 28}, rs=8, p=8, nmode=351;

// Kriging basis computed by emuInit
// Sizes of each basis will be peta[ee] and m[ee]
static double KrigBasis[2][28][111];

// Need to combine some things into a big thing.
static double beta[2][28][8];
static double w[2][28][111];
static double lamws[2][28];
static double lamz[2][28];

// Initialization to compute the kriging basis for the two parts
void emuInit() {
   
    int ee, i, j, k, l;
    double cov;
    gsl_matrix *SigmaSim;
    gsl_vector *b;

    // Because of the structure of this emu, I need to do this horrible part first.
    // Fill in the correlation lenghths
    for(i=0; i<7; i++) {
        for(j=0; j<8; j++) {
            beta[0][i][j] = beta1[i][j];
        }
    }
    
    for(i=0; i<28; i++) {
        for(j=0; j<8; j++) {
            beta[1][i][j] = beta2[i][j];
        }
    }
    
    // Fill in the PC weights
    for(i=0; i<7; i++) {
        for(j=0; j<111; j++) {
            w[0][i][j] = w1[i][j];
        }
    }
    
    for(i=0; i<28; i++) {
        for(j=0; j<36; j++) {
            w[1][i][j] = w2[i][j];
        }
    }
    
    // Fill in the precisions
    for(i=0; i<7; i++) {
        lamws[0][i] = lamws1[i];
        lamz[0][i] = lamz1[i];
    }
    
    for(i=0; i<28; i++) {
        lamws[1][i] = lamws2[i];
        lamz[1][i] = lamz2[i];
    }
    
    // This emu has two parts: one that uses all m[0] of the data for the the first peta[0] components
    // and another that uses only the m[1] complete data for the next peta[1] components.
    for(ee=0; ee<2; ee++) {
        
        // Allocate some stuff
        SigmaSim = gsl_matrix_alloc(m[ee], m[ee]);
        b = gsl_vector_alloc(m[ee]);
        
        // Loop over the basis
        for(i=0; i<peta[ee]; i++) {
            
            // Loop over the number of simulations
            for(j=0; j<m[ee]; j++) {
                
                // Diagonal term
                gsl_matrix_set(SigmaSim, j, j, (1.0/lamz[ee][i]) + (1.0/lamws[ee][i]));
                
                // Off-diagonals
                for(k=0; k<j; k++) {
                    
                    // compute the covariance
                    cov = 0.0;
                    for(l=0; l<p; l++) {
                        cov -= beta[ee][i][l]*pow(x[j][l] - x[k][l], 2.0);
                    } // for(l=0; l<p; l++)
                    cov = exp(cov) / lamz[ee][i];
                    
                    // put the covariance where it belongs
                    gsl_matrix_set(SigmaSim, j, k, cov);
                    gsl_matrix_set(SigmaSim, k, j, cov);
                    
                } // for(k=0; k<j; k++)
                
                // Vector for the PC weights
                gsl_vector_set(b, j, w[ee][i][j]);
                
            } // for(j=0; j<m[ee]; j++)
            
            // Cholesky and solve
            gsl_linalg_cholesky_decomp(SigmaSim);
            gsl_linalg_cholesky_svx(SigmaSim, b);
            
            // Put b where it belongs in the Kriging basis
            for(j=0; j<m[ee]; j++) {
                KrigBasis[ee][i][j] = gsl_vector_get(b, j);
            }
            
        } // for(i=0; i<peta[ee]; i++)
        
        // Clean this up
        gsl_matrix_free(SigmaSim);
        gsl_vector_free(b);
        
    } // for(ee=0; ee<2; ee+)
    
} // emuInit()

// Actual emulation
void emu(double *xstar, double *ystar) {
    
    static double inited=0;
    int ee, i, j, k;
    double wstar[peta[0]+peta[1]];
    double Sigmastar[2][peta[1]][m[0]];
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
    
    /* Something is in there.
    // Can I see KrigBasis with stuff in it?
    for(k=0; k<2; k++) {
        for(i=0; i<peta[k]; i++) {
            for(j=0; j<m[k]; j++) {
                printf("%f ", KrigBasis[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
     */
    
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
    if((xstar[p] < z[0]) || (xstar[p] > z[rs-1])) {
        printf("z must be between %f and %f.\n", z[0], z[rs-1]);
        printf("%f\n", xstar[p]-z[rs-1]);
        exit(1);
    }
    
    // Standardize the inputs
    for(i=0; i<p; i++) {
        xstarstd[i] = (xstar[i] - xmin[i]) / xrange[i];
        //printf("%f %f\n", xstar[i], xstarstd[i]);
    }
    
    /* Yes.
    // Are we getting through
    for(i=0; i<p; i++) {
        printf("%f ", xstarstd[i]);
    }
    printf("\n");
     */
    
    // compute the covariances between the new input and sims for all the PCs.
    for(ee=0; ee<2; ee++) {
        for(i=0; i<peta[ee]; i++) {
            for(j=0; j<m[ee]; j++) {
                logc = 0.0;
                for(k=0; k<p; k++) {
                    logc -= beta[ee][i][k]*pow(x[j][k] - xstarstd[k], 2.0);
                }
                Sigmastar[ee][i][j] = exp(logc) / lamz[ee][i];
            }
        }
    }
    
    // Compute wstar for the first chunk.
    for(i=0; i<peta[0]; i++) {
        wstar[i]=0.0;
        for(j=0; j<m[0]; j++) {
            wstar[i] += Sigmastar[0][i][j] * KrigBasis[0][i][j];
        }
    }
    // Compute wstar for the second chunk.
    for(i=0; i<peta[1]; i++) {
        wstar[peta[0]+i]=0.0;
        for(j=0; j<m[1]; j++) {
            wstar[peta[0]+i] += Sigmastar[1][i][j] * KrigBasis[1][i][j];
        }
    }
    
    /*
    for(i=0; i<peta[0]+peta[1]; i++) {
        printf("%f\n", wstar[i]);
    }
     */
    
    // Compute ystar, the new output
    for(i=0; i<neta; i++) {
        ystaremu[i] = 0.0;
        for(j=0; j<peta[0]+peta[1]; j++) {
            ystaremu[i] += K[i][j]*wstar[j];
        }
        ystaremu[i] = ystaremu[i]*sd + mean[i];
        
        //printf("%f\n", ystaremu[i]);
        
        // Convert to P(k)
        //ystaremu[i] = ystaremu[i] - 1.5*log10(mode[i % nmode]);
        //ystaremu[i] = 2*M_PI*M_PI*pow(10, ystaremu[i]);
    }
    
    
    
    // Interpolate to the desired redshift
    // Natural cubic spline interpolation over z.
    
    // First check to see if the requested z is one of the training z.
    zmatch = -1;
    for(i=0; i<rs; i++) {
        if(xstar[p] == z[i]) {
            zmatch = rs-i-1;
        }
    }
    
    // z doesn't match a training z, interpolate
    if(zmatch == -1) {
        for(i=0; i<nmode; i++) {
            for(j=0; j<rs; j++) {
                ybyz[rs-j-1] = ystaremu[j*nmode+i];
            }
            gsl_spline_init(zinterp, z, ybyz, rs);
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
    // Intensive use of the emulator should probably use something smarter.
    
    // Parameter order
    // '\omega_m'   '\omega_b'   '\sigma_8'   'h'   'n_s'   'w_0'   'fw'   '\omega_{\nu}'   'z'
    
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
    
    /*
    emu(xstar, ystar);
    for(i=0; i<351; i++) {
        printf("%f %f \n", mode[i], ystar[i]);
    }
    printf("\n");
    */
}
