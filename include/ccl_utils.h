#pragma once

/**
 * Compute bin edges of N-1 linearly spaced bins on the interval [xmin,xmax]
 * @param xmin minimum value of spacing
 * @param xmax maximum value of spacing
 * @param N number of bins plus one (number of bin edges)
 * @return x, bin edges in range [xmin, xmax]
 */
double * ccl_linear_spacing(double xmin, double xmax, int N);


/**
 * Compute bin edges of N-1 logarithmically spaced bins on the interval [xmin,xmax]
 * @param xmin minimum value of spacing
 * @param xmax maximum value of spacing
 * @param N number of bins plus one (number of bin edges)
 * @return x, bin edges in range [xmin, xmax]
 */
double * ccl_log_spacing(double xmin, double xmax, int N);

