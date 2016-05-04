#pragma once

double * ccl_linear_spacing(double xmin, double xmax, double dx, int * N);
//Returns array of linearly-spaced values between xmin and xmax
//with linear spacing dx. Number of elements returned in N.

double * ccl_log_spacing(double xmin, double xmax, int N);
//Returns array of N logarithmically-spaced values between xmin and xmax
