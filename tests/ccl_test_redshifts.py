import numpy as np
from numpy.testing import assert_allclose, run_module_suite
import numpy.testing
import pyccl as ccl

# Set tolerances
TOLERANCE = 1e-4

def test_redshift_analytic():
    """
    Compare the redshift functions to an analytic toy case.
    """
    # Redshift input
    z_lst = [0., 0.5, 1., 1.5, 2.]

    # p(z) function for which we can calculate and analytic dNdz_tomog
    # when pairs with the toy dndz_ana below
    def pz_ana(z_ph, z_s, args):
        return (np.exp(- (z_ph - z_s)**2. / 2. / 0.1**2) / np.sqrt(2.
                * np.pi) / 0.1)

    # PhotoZFunction class
    PZ_ana = ccl.PhotoZFunction(pz_ana)
        
    # Introduce an unrealistic, simple true function for dNdz for which
    # we can calculate dNdz_tomog analytically for comparison.
    def dndz_ana(z, args):
        if ((z>=0.) and (z<=1.0)): 
            return 1.0
        else:
            return 0. 
	
    # import math erf and erfc functions:
    from math import erf, erfc
    
    # Return the analytic result for dNdz_tomog using 
    # dndz_ana with a Gaussian p(z,z'), to which we compare.
    
    # This is obtained by analytically evaluating (in pseudo-latex):
    # \frac{ dndz_ana(z) \int_{zpmin}^{zpmax} dz_p pz_ana(z_p, z_s)}
    # {\int_{0}^{1} dz_s dndz_ana(z_s) \int_{zpmin}^{zpmax} dz_p 
    # pz_ana(z_p, z_s)}
    # (which we did using Wolfram Mathematica)
    
    def dNdz_tomog_analytic(z, sigz, zmin, zmax):
        if ( (z>=0.) and (z<=1.0) ):
            return (erf((z-zmin) / np.sqrt(2.)/sigz) - erf((z-zmax)/
                    np.sqrt(2.)/sigz)) / (-1. + (-np.exp(-(zmax-1)**2 
                    / 2. / sigz**2) + np.exp(-zmax**2 / 2. / sigz**2) + 
                    np.exp(-(zmin-1)**2 / 2. / sigz**2) - np.exp(
                    -zmin**2 / 2 /sigz**2)) * np.sqrt(2. / np.pi)*sigz 
                    + erf( (zmax-1) / np.sqrt(2.) /  sigz) + erfc( 
                    (zmin-1.)/np.sqrt(2.)/sigz) + zmax * (erf(zmax/
                    np.sqrt(2.)/sigz) - erf( (zmax-1.) / np.sqrt(2.)/
                    sigz)) + zmin * ( erf( (zmin-1.)/np.sqrt(2.)
		            / sigz) - erf(zmin/np.sqrt(2.)/sigz)))
        else:
            return 0.
        
    # dNdzFunction class
    dNdZ_ana = ccl.dNdzFunction(dndz_ana)
    
    # Check that for the analytic case introduced above, we get the 
    # correct value.
    zmin = 0.
    zmax = 1.
    # math erf funcs are not vectorized so loop over z values
    for z in z_lst:
        assert_allclose(ccl.dNdz_tomog(z, zmin, zmax, PZ_ana, 
		               dNdZ_ana), dNdz_tomog_analytic(z, 0.1, zmin, 
		               zmax), rtol=TOLERANCE)       
		            
def test_redshift_numerical():
    """
    Compare the redshift functions to a high precision integral.
    """
    # Redshift input
    z_lst = [0., 0.5, 1., 1.5, 2.]
                
     # p(z) function for dNdz_tomog
    def pz1(z_ph, z_s, args):
        return np.exp(- (z_ph - z_s)**2. / 2.)

    # Lambda function p(z) function for dNdz_tomog
    pz2 = lambda z_ph, z_s, args: np.exp(-(z_ph - z_s)**2. / 2.)
    
    # Set up a function equivalent to the PhotoZGaussian
    def pz3(z_ph, z_s, sigz):
        sig = sigz*(1.+ z_s)
        return (np.exp(- (z_ph - z_s)**2. / 2. / sig**2) / np.sqrt(2.
		        *np.pi) / sig)

    # PhotoZFunction classes
    PZ1 = ccl.PhotoZFunction(pz1)
    PZ2 = ccl.PhotoZFunction(pz2)
    PZ3 = ccl.PhotoZGaussian(sigma_z0=0.1)
    
    # dNdz (in terms of true redshift) function for dNdz_tomog
    def dndz1(z, args):
        return z**1.24 * np.exp(- (z / 0.51)**1.01)
    
    # dNdzFunction classes
    dNdZ1 = ccl.dNdzFunction(dndz1)
    dNdZ2 = ccl.dNdzSmail(alpha = 1.24, beta = 1.01, z0 = 0.51)
    
    # Do the integral in question directly in numpy at high precision
    zmin = 0.
    zmax = 1.
    zp = np.linspace(zmin, zmax, 10000)
    zs = np.linspace(0., 5., 10000) # Assume any dNdz does not extend 
    # above z=5
    denom_zp_1 =np.asarray([np.trapz(pz1(zp, z, []), zp) for z in zs])
    denom_zp_2 =np.asarray([np.trapz(pz2(zp, z, []), zp) for z in zs])
    denom_zp_3 =np.asarray([np.trapz(pz3(zp, z, 0.1), zp) for z in zs])
    np_dndz_1 = ([ dndz1(z, []) * np.trapz(pz1(zp, z, []), zp) / 
                 np.trapz(dndz1(zs, []) * denom_zp_1, zs) for z in 
                 z_lst])
    np_dndz_2 = ([ dndz1(z, []) * np.trapz(pz2(zp, z, []), zp) / 
                 np.trapz(dndz1(zs, []) * denom_zp_2, zs) for z in 
                 z_lst])
    np_dndz_3 = ([ dndz1(z, []) * np.trapz(pz3(zp, z, 0.1), zp) / 
                 np.trapz(dndz1(zs, []) * denom_zp_3, zs) for z in 
                 z_lst])
    
    # Check that for the analytic case introduced above, we get the 
    # correct value.
    for i in range(0, len(z_lst)):
        assert_allclose(ccl.dNdz_tomog(z_lst[i], zmin, zmax, PZ1, 
		               dNdZ1), np_dndz_1[i], rtol=TOLERANCE) 
        assert_allclose(ccl.dNdz_tomog(z_lst[i], zmin, zmax, PZ1, 
		               dNdZ2), np_dndz_1[i], rtol=TOLERANCE)  
        assert_allclose(ccl.dNdz_tomog(z_lst[i], zmin, zmax, PZ2, 
		               dNdZ1), np_dndz_2[i], rtol=TOLERANCE) 
        assert_allclose(ccl.dNdz_tomog(z_lst[i], zmin, zmax, PZ2, 
		               dNdZ2), np_dndz_2[i], rtol=TOLERANCE)  
        assert_allclose(ccl.dNdz_tomog(z_lst[i], zmin, zmax, PZ3, 
		               dNdZ1), np_dndz_3[i], rtol=TOLERANCE) 
        assert_allclose(ccl.dNdz_tomog(z_lst[i], zmin, zmax, PZ3, 
		               dNdZ2), np_dndz_3[i], rtol=TOLERANCE)  

if __name__ == "__main__":
    run_module_suite()
