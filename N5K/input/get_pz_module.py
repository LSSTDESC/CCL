import numpy as np
from scipy.stats import norm
from copy import deepcopy
from scipy.stats import norm
import scipy.interpolate
import sys
import matplotlib.pyplot as plt

""" This script originally by Markus Michael Rau, modified by Danielle Leonard. """


class PhotoZ_core(object):
    """Defines p(zp | z) of the core distribution (i.e. no outliers)
    """

    def __init__(self, zp_support, zbias, sigma_z):
        self.zp_support = zp_support
        self.zbias = zbias
        self.sigma_z = sigma_z

    def get_pdf_given(self, z):
        assert np.isscalar(z)
        rv = norm()
        scale = self.sigma_z * (1. + z)
        loc = z - self.zbias
        return rv.pdf((self.zp_support - loc) / scale) / scale


class PhotoZ_outlier(object):

    def __init__(self, zp_support, z_cat, sigma_cat):
        self.z_cat = z_cat
        self.sigma_cat = sigma_cat
        self.zp_support = zp_support

    def get_pdf_given(self, z):
        assert np.isscalar(z)
        rv = norm()
        return rv.pdf((self.zp_support - self.z_cat) / self.sigma_cat) / self.sigma_cat


class PhotoZ_outlier_position(object):
    def __init__(self, F_cat, delta_z_cat, z_cat_position):
        assert np.isscalar(F_cat)
        self.delta_z_cat = delta_z_cat
        self.z_cat_position = z_cat_position
        self.F_cat = F_cat

    def get_weight_given(self, z):
        assert np.isscalar(z)
        if (z > self.z_cat_position - self.delta_z_cat) and (z < self.z_cat_position + self.delta_z_cat):
            return self.F_cat
        else:
            return 0.


class LymanBalmerConfusion(object):
    def __init__(self, zp_support):
        self.outlier_pdf = PhotoZ_outlier(zp_support, z_cat=3.75, sigma_cat=0.25)

    def get_pdf_given(self, z):
        # Note this pdf is normalized to F_cat or zero!!!
        assert np.isscalar(z)
        outlier_pdf = self.outlier_pdf.get_pdf_given(z)
        return outlier_pdf


class LymanAlphaBalmerConfusion(object):
    def __init__(self, zp_support):
        self.outlier_pdf = PhotoZ_outlier(zp_support, z_cat=0.5, sigma_cat=0.25)

    def get_pdf_given(self, z):
        # Note this pdf is normalized to F_cat or zero!!!
        assert np.isscalar(z)
        outlier_pdf = self.outlier_pdf.get_pdf_given(z)
        return outlier_pdf


class CoreAndOutlier(object):
    def __init__(self, zp_support, zbias, sigma_z, F_cat_lyman_balmer, F_cat_lyman_alpha_balmer):
        self.zp_support = zp_support
        # DL: get the basic pdf with a redshift bias and variance.
        self.core = PhotoZ_core(zp_support, zbias, sigma_z)

        self.pos_lyman_balmer = PhotoZ_outlier_position(F_cat_lyman_balmer,
                                                        delta_z_cat=0.25, z_cat_position=0.25)
        self.lyman_balmer = LymanBalmerConfusion(zp_support)

        self.pos_lyman_alpha_balmer = PhotoZ_outlier_position(F_cat_lyman_alpha_balmer,
                                                              delta_z_cat=0.25, z_cat_position=3.)
        self.lyman_alpha_balmer = LymanAlphaBalmerConfusion(zp_support)

    def get_pdf_given(self, z):
        assert np.isscalar(z)
        # DL: Get the weights to give to each of the confusion elements
        weight_lyman_balmer = self.pos_lyman_balmer.get_weight_given(z)
        weight_lyman_alpha_balmer = self.pos_lyman_alpha_balmer.get_weight_given(z)
        weight_core = 1.0 - weight_lyman_balmer - weight_lyman_alpha_balmer
        
        core_pdf = self.core.get_pdf_given(z)
        core_pdf = core_pdf / np.trapz(core_pdf, self.zp_support)

        lyman_balmer_outliers = self.lyman_balmer.get_pdf_given(z)
        lyman_balmer_outliers = lyman_balmer_outliers / np.trapz(lyman_balmer_outliers, self.zp_support)

        lyman_alpha_balmer_outliers = self.lyman_alpha_balmer.get_pdf_given(z)
        lyman_alpha_balmer_outliers = lyman_alpha_balmer_outliers / np.trapz(lyman_alpha_balmer_outliers,
                                                                             self.zp_support)

        cond_pdf = weight_core * core_pdf + weight_lyman_balmer * lyman_balmer_outliers + weight_lyman_alpha_balmer * lyman_alpha_balmer_outliers
        return cond_pdf


class SmailZ(object):
    def __init__(self, z_support, z0, gamma, alpha):
        pdf = z_support ** alpha * np.exp(-(z_support / z0) ** (gamma))
        self.z_support = z_support
        self.pdf = pdf / np.trapz(pdf, z_support)

    def get_pdf(self):
        return self.pdf

    def get_pdf_convoled(self, filter_list):
        output_tomo_list = []
        for el in filter_list:
            output_tomo_list.append(el * self.pdf)
        output_tomo_list = np.array(output_tomo_list).T
        output_tomo_list = np.column_stack((self.z_support, output_tomo_list))
        return output_tomo_list

class HistoZ(object):
    def __init__(self, z_support, dNdz):
        pdf = dNdz
        self.z_support = z_support
        self.pdf = pdf / np.trapz(pdf, z_support)

    def get_pdf(self):
        return self.pdf

    def get_pdf_convoled(self, filter_list):
        output_tomo_list = []
        for el in filter_list:
            output_tomo_list.append(el * self.pdf)
        output_tomo_list = np.array(output_tomo_list).T
        output_tomo_list = np.column_stack((self.z_support, output_tomo_list))
        return output_tomo_list

class NakajimaZ(object):
    def __init__(self, z_support, zs, alpha):
        pdf = (z_support / zs) ** (alpha - 1.) * np.exp(-0.5 * (z_support / zs) ** 2)
        self.z_support = z_support
        self.pdf = pdf / np.trapz(pdf, z_support)

    def get_pdf(self):
        return self.pdf

    def get_pdf_convoled(self, filter_list):
        output_tomo_list = []
        for el in filter_list:
            output_tomo_list.append(el * self.pdf)
        output_tomo_list = np.array(output_tomo_list).T
        output_tomo_list = np.column_stack((self.z_support, output_tomo_list))
        return output_tomo_list


class PhotozModel(object):
    def __init__(self, pdf_z, pdf_zphot_given_z, filters):
        self.pdf_zphot_given_z = pdf_zphot_given_z
        self.pdf_z = pdf_z
        self.filter_list = filters

    def get_pdf(self):
		
        # Get the pdf_zphot_given_z stuff done here 
        # this is not dependent on the filter so there's no need to
        # do it each time.
        z_support = self.pdf_z.z_support
        zp_support = self.pdf_zphot_given_z.zp_support
        zphot_given_z = np.zeros((len(zp_support), len(z_support)))
        
        for i in range(0,len(z_support)):
            zphot_given_z[:, i] = self.pdf_zphot_given_z.get_pdf_given(z_support[i])
		
        tomo_collect = []
        for el in self.filter_list:
            tomo_collect.append(self.get_pdf_tomo(el, zphot_given_z))

        tomo_collect = np.array(tomo_collect).T
        return tomo_collect

    def get_pdf_tomo(self, filter, zphot_given_z):
        z_support = self.pdf_z.z_support
        zp_support = self.pdf_zphot_given_z.zp_support
        z_pdf = self.pdf_z.get_pdf()
        pdf_joint = np.zeros((len(zp_support), len(z_support)))
        for i in range(len(z_support)):
            pdf_joint[:, i] = zphot_given_z[:,i] * z_pdf[i] * filter[i]
        
        pdf_zp = np.zeros((len(zp_support),))
        for i in range(len(zp_support)):
            pdf_zp[i] = np.trapz(pdf_joint[i, :], z_support)

        return pdf_zp


def get_pz_func(zbias_cl, sigma_z_cl, F_cat_lyman_balmer_cl, 
                F_cat_lyman_alpha_balmer_cl, zbias_src, sigma_z_src, 
                F_cat_lyman_balmer_src, F_cat_lyman_alpha_balmer_src):

    # Main function to get the photo-z distributions.

    # Leave these print statements in if you want to see what parameters you passed in.
    print(zbias_cl, sigma_z_cl, F_cat_lyman_balmer_cl,  F_cat_lyman_alpha_balmer_cl)
    print(zbias_src, sigma_z_src, F_cat_lyman_balmer_src, F_cat_lyman_alpha_balmer_src)

    # Uncomment this block if using Y10.
    # Edit filepaths to where you have unpacked the SRD tarball
    # SRD LSST Y10 direct from file, sources
    (z_low_src, zp_temp_src, z_high_src, pdf_temp_src) = np.loadtxt('/data/danielle/research/iaxphoto/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=1.100000e-01_beta=6.800000e-01_Y10_source', unpack=True)     
    nbin_src=3
    #fname_zp_out_src = './dNdz_srcs_nbins'+str(nbin_src)+'_quarterwidth_sigz0.01.dat'
    fname_zp_out_src = './dNdz_srcs_nbins'+str(nbin_src)+'_fiducial_cutbins.dat'
    # SRD LSST Y10 direct from file, lenses
    (z_low_cl, zp_temp_cl, z_high_cl, pdf_temp_cl) = np.loadtxt('/data/danielle/research/iaxphoto/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=2.800000e-01_beta=9.000000e-01_Y10_lens', unpack=True) 
    nbin_cl=6
    #fname_zp_out_cl = './dNdz_clust_nbins'+str(nbin_cl)+'_quarterwidth_sigz0.006.dat'    
    fname_zp_out_cl = './dNdz_clust_nbins'+str(nbin_cl)+'_fiducial_cutbins.dat'  
    
    # Uncomment this block if using Y1
    # Edit filepaths to where you have unpacked the SRD tarball 
    # SRD LSST Y1 direct from file sources
    #(z_low_cl, zp_temp_cl, z_high_cl, pdf_temp_cl) = np.loadtxt('/data/danielle/research/iaxphoto/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=1.300000e-01_beta=7.800000e-01_Y1_source', unpack=True) 
    #fname_zp_out_cl = './dNdz_srcs_LSSTSRD_Y1.dat'    
    #nbin_cl=5 
    # SRD LSST Y1 direct from file lenses
    #(z_low_src, zp_temp_src, z_high_src, pdf_temp_src) = np.loadtxt('/data/danielle/research/iaxphoto/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=2.600000e-01_beta=9.400000e-01_Y1_lens', unpack=True)
    #fname_zp_out_src = './dNdz_clust_LSSTSRD_Y1.dat'    
    #nbin_src=5
    
    #Interpolate and do higher z sampling
    interp_dNdz_src = scipy.interpolate.interp1d(zp_temp_src, pdf_temp_src)
    z_support_src = np.linspace(zp_temp_src[0], zp_temp_src[-1], 2000)
    pdf_smail_src = interp_dNdz_src(z_support_src)
    zp_support_src = z_support_src
    
    interp_dNdz_cl = scipy.interpolate.interp1d(zp_temp_cl, pdf_temp_cl)
    z_support_cl = np.linspace(zp_temp_cl[0], zp_temp_cl[-1], 2000)
    pdf_smail_cl = interp_dNdz_cl(z_support_cl)
    zp_support_cl = z_support_cl
    
    pdf_z_src = HistoZ(z_support_src, pdf_smail_src)
    pdf_z_cl = HistoZ(z_support_cl, pdf_smail_cl)

    # For sources, we want equal numbers of galaxies in each bin
    CDF = [ np.trapz(pdf_smail_src[0:i], z_support_src[0:i]) for i in range(len(pdf_smail_src))]
    
    # CHANGE THIS to change number of bins
    #cuts_src = [0., 0.2, 0.4, 0.6, 0.8] #fiducial
    #cuts_src = [0., 0.1, 0.2, 0.3, 0.4, 0.5] # half width
    #cuts_src = [0., 0.05, 0.1, 0.15, 0.2, 0.25] # quarter width
    #cuts_src = [0., 0.2, 0.4, 0.6, 0.8] # cut bins, 4 bins
    cuts_src = [0., 0.2, 0.4, 0.6] # cut bins, 3 bins

    # Indices of bin edges:
    bin_edges_src= [next(j[0] for j in enumerate(CDF) if j[1]>cuts_src[i]) for i in range(len(cuts_src))]
    #bin_edges_src.append(len(z_support_src)-1)
    
    # For lenses, we just use equal-spaced bins.

    # Uncomment this block for Y10

    # CHANGE THIS to change number of bins
    #cuts_cl = [0.2 + 0.1*i for i in range(0,11)] # fiducial
    #cuts_cl = [0.2 + 0.05*i for i in range(0,11)] # half width
    #cuts_cl = [0.2 + 0.025*i for i in range(0,11)] # quarter width
    #cuts_cl = [0.2 + 0.1*i for i in range(0,9)] # cut bins, 8 bins
    cuts_cl = [0.2 + 0.1*i for i in range(0,7)] # cut bins, 6 bins
    print(cuts_cl)

    bin_edges_cl= [next(j[0] for j in enumerate(z_support_cl) if j[1]>cuts_cl[i]) for i in range(len(cuts_cl))]
    #bin_edges_cl.append(len(z_support_cl)-1)

    # Uncomment this block for Y1
    #cuts_cl = [0.2 + 0.2*i for i in range(0,6)]
    #bin_edges_cl= [next(j[0] for j in enumerate(z_support_cl) if j[1]>cuts_cl[i]) for i in range(len(cuts_cl))]
    #bin_edges_cl.append(len(z_support_cl)-1)

    # The following are the bin functions to bin the true redshift distribution
    # For the SRD, we want to use top-hats here.
    # note that these have to be normalized

    bin1_cl = [1. if (z>z_support_cl[bin_edges_cl[0]] and z<=z_support_cl[bin_edges_cl[1]]) else 0. for z in z_support_cl]
    bin1_cl = np.asarray(bin1_cl)
    bin2_cl = [1. if (z>z_support_cl[bin_edges_cl[1]] and z<=z_support_cl[bin_edges_cl[2]]) else 0. for z in z_support_cl]
    bin2_cl = np.asarray(bin2_cl)
    bin3_cl = [1. if (z>z_support_cl[bin_edges_cl[2]] and z<=z_support_cl[bin_edges_cl[3]]) else 0. for z in z_support_cl]
    bin3_cl = np.asarray(bin3_cl)
    bin4_cl = [1. if (z>z_support_cl[bin_edges_cl[3]] and z<=z_support_cl[bin_edges_cl[4]]) else 0. for z in z_support_cl]
    bin4_cl = np.asarray(bin4_cl)
    bin5_cl = [1. if (z>z_support_cl[bin_edges_cl[4]] and z<=z_support_cl[bin_edges_cl[5]]) else 0. for z in z_support_cl]
    bin5_cl = np.asarray(bin5_cl)

    # If you are using Y10, need to uncomment the below:
    # CHANGE THIS to change number of bins

    bin6_cl = [1. if (z>z_support_cl[bin_edges_cl[5]] and z<=z_support_cl[bin_edges_cl[6]]) else 0. for z in z_support_cl]
    bin6_cl = np.asarray(bin6_cl)
    #bin7_cl = [1. if (z>z_support_cl[bin_edges_cl[6]] and z<=z_support_cl[bin_edges_cl[7]]) else 0. for z in z_support_cl]
    #bin7_cl = np.asarray(bin7_cl)
    #bin8_cl = [1. if (z>z_support_cl[bin_edges_cl[7]] and z<=z_support_cl[bin_edges_cl[8]]) else 0. for z in z_support_cl]
    #bin8_cl = np.asarray(bin8_cl)
    #bin9_cl = [1. if (z>z_support_cl[bin_edges_cl[8]] and z<=z_support_cl[bin_edges_cl[9]]) else 0. for z in z_support_cl]
    #bin9_cl = np.asarray(bin9_cl)
    #bin10_cl = [1. if (z>z_support_cl[bin_edges_cl[9]] and z<=z_support_cl[bin_edges_cl[10]]) else 0. for z in z_support_cl]
    #bin10_cl = np.asarray(bin10_cl)

    # CHNAGE THIS to change number of bins
    tomo_list_cl = [bin1_cl/np.trapz(bin1_cl, z_support_cl),
                 bin2_cl/np.trapz(bin2_cl, z_support_cl),
                 bin3_cl/np.trapz(bin3_cl, z_support_cl),
                 bin4_cl/np.trapz(bin4_cl, z_support_cl),
                 bin5_cl/np.trapz(bin5_cl, z_support_cl), 
                 bin6_cl/np.trapz(bin6_cl, z_support_cl)]#,
                 #bin7_cl/np.trapz(bin7_cl, z_support_cl),
                 #bin8_cl/np.trapz(bin8_cl, z_support_cl)]#,
                 #bin9_cl/np.trapz(bin9_cl, z_support_cl),
                 #bin10_cl/np.trapz(bin10_cl, z_support_cl)]
                 # If using Y10, need to uncomment 6-10 and include them in the list (remove ])
    
    # CHANGE THIS to change number of bins
    # These are the bin functions for the sources - same number of bins for Y1 and Y10 so don't need to change this.
    bin1_src = [1. if (z>z_support_src[bin_edges_src[0]] and z<=z_support_src[bin_edges_src[1]]) else 0. for z in z_support_src]
    bin1_src = np.asarray(bin1_src)
    bin2_src = [1. if (z>z_support_src[bin_edges_src[1]] and z<=z_support_src[bin_edges_src[2]]) else 0. for z in z_support_src]
    bin2_src = np.asarray(bin2_src)
    bin3_src = [1. if (z>z_support_src[bin_edges_src[2]] and z<=z_support_src[bin_edges_src[3]]) else 0. for z in z_support_src]
    bin3_src = np.asarray(bin3_src)
    #bin4_src = [1. if (z>z_support_src[bin_edges_src[3]] and z<=z_support_src[bin_edges_src[4]]) else 0. for z in z_support_src]
    #bin4_src = np.asarray(bin4_src)
    #bin5_src = [1. if (z>z_support_src[bin_edges_src[4]] and z<=z_support_src[bin_edges_src[5]]) else 0. for z in z_support_src]
    #bin5_src = np.asarray(bin5_src)
    tomo_list_src = [bin1_src/np.trapz(bin1_src, z_support_src),
                 bin2_src/np.trapz(bin2_src, z_support_src),
                 bin3_src/np.trapz(bin3_src, z_support_src)]#,
                 #bin4_src/np.trapz(bin4_src, z_support_src)]#,
                 #bin5_src/np.trapz(bin5_src, z_support_src)]                          

    # Now pass all this stuff to the functions that setup the photo-z distributions
        
    coreandoutlier_cl= CoreAndOutlier(zp_support_cl, zbias=zbias_cl, sigma_z=sigma_z_cl, F_cat_lyman_balmer=F_cat_lyman_balmer_cl, F_cat_lyman_alpha_balmer=F_cat_lyman_alpha_balmer_cl)   
    coreandoutlier_src= CoreAndOutlier(zp_support_src, zbias=zbias_src, sigma_z=sigma_z_src, F_cat_lyman_balmer=F_cat_lyman_balmer_src, F_cat_lyman_alpha_balmer=F_cat_lyman_alpha_balmer_src)   
      
    photoz_model_cl = PhotozModel(pdf_z_cl, coreandoutlier_cl, tomo_list_cl)
    photoz_model_src = PhotozModel(pdf_z_src, coreandoutlier_src, tomo_list_src)

    # This returns dNdz_p by convolving with the photo-z model and with
    # the fiters defined by tomo_list (top hat, Gaussian etc)
    pdf_zp_cl = photoz_model_cl.get_pdf()
    pdf_zp_src = photoz_model_src.get_pdf()

    # Output - set these file names at the beginning of the function.
    output_zp_cl = np.column_stack((zp_support_cl, pdf_zp_cl))
    np.savetxt(X=output_zp_cl, fname=fname_zp_out_cl)
    output_zp_src = np.column_stack((zp_support_src, pdf_zp_src))
    np.savetxt(X=output_zp_src, fname=fname_zp_out_src)


    return (zp_support_cl, pdf_zp_cl, zp_support_src, pdf_zp_src)

