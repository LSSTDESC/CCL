import numpy as np
import os, subprocess, glob, random
import pyccl as ccl


def build_data_dict(stats_arr, prefix):
    """
    Build a data dictionary with columns named according to (k,z) bins, a
    threshold value, and some prefix.

    Assumes that stats_arr has shape: (N_samp, N_thres, N_z, N_kbins)
    """
    # Get no. of points in each dimension.
    N_sam, N_thres, N_z, N_kbins = stats_arr.shape

    # Create dictionary with column names that can be used by ColumnDataSource
    data_dict = {}
    for n in range(N_thres):
        for j in range(N_z):
            for m in range(N_kbins):
                key = "tot_%s_h%d_k%d_z%d" % (prefix, n+1, m+1, j+1)
                data_dict[key] = stats_arr[:,n, j, m]
    return data_dict


def save_hypercube(fname, sample_points):
    """
    Save a Latin hypercube to disk as a text file. Parameter columns will be
    ordered alphabetically by name.

    Parameters
    ----------
    fname : str
        Filename for output file.
    sample_points : dict
        Dictionary containing parameter names (keys) and array of parameter
        values for all sample points (values).
    """
    # Get parameter names and build header
    pnames = sample_points.keys()
    pnames.sort()
    hdr = " ".join(pnames)

    # Build array
    dat = np.column_stack([sample_points[p] for p in pnames])
    np.savetxt(fname, dat, header=hdr, fmt="%4.4e")
    print("Saved hypercube to '%s'." % fname)


def load_hypercube(fname):
    """
    Load a Latin hypercube from disk. Parameter columns will be ordered
    alphabetically by name.

    Parameters
    ----------
    fname : str
        Filename to read sample points from.

    Returns
    -------
    sample_points : dict
        Dictionary containing parameter names (keys) and array of parameter
        values for all sample points (values).
    """
    # Get header
    f = open(fname, 'r')
    hdr = f.readline()[2:-1].split(" ")
    f.close()

    # Load data
    dat = np.loadtxt(fname).T

    # Build dict
    sample_points = {}
    for i in range(len(hdr)):
        sample_points[hdr[i]] = dat[i]
    return sample_points


def generate_latin_hypercube(samples, param_dict, class_root, seed=10):
    """
    Generate a Latin hypercube for a given set of parameters.

    Parameters
    ----------
    samples : int
        Number of samples points to draw from the hypercube.

    param_dict : dict
        Dictionary containing parameter names (keys) and tuple with min/max.
        values (values).

    class_root : str
        Path to directory containing 'class' executable.

    seed : int, optional
        Random seed to use when sampling from hypercube.

    Returns
    -------
    sample_points : dict
        Dictionary containing parameter names (keys) and array of parameter
        values for all sample points (values).
    """
    # Set random seed
    random.seed(seed)

    # Create dictionary to hold sampled parameter values
    sample_points = {}
    for key in param_dict.keys():
        sample_points[key] = np.zeros(samples)
    Ndim = len(param_dict.keys())
    pnames = [key for key in param_dict.keys()]

    # List of indices for each dimension
    l = [range(samples) for j in range(Ndim)]

    # Generate samples until there are no indices left to choose
    for i in range(samples):

        # Randomly choose index and then remove the number that was chosen
        # (Latin hypercubes require at most one item per row and column)
        for j, p in enumerate(pnames):
            pmin, pmax = param_dict[p]
            idx = random.choice(l[j])

            # Get value at this sample point (add 0.5 to idx get bin centroid)
            sample_points[p][i] = pmin + (pmax - pmin) \
                                * (idx + 0.5) / float(samples)
            l[j].remove(idx) # Remove choice from list (sampling w/o replacement)

    return sample_points


def generate_ccl_pspec(sample_points, root, class_data_root, zvals,
                       default_params=None, nonlin=False, mode='std'):
    """
    Generate linear and non-linear power spectra using CCL, for a set of
    points in cosmological parameter space and redshift.

    Parameters
    ----------
    sample_points : dict
        Dictionary containing parameter names (keys) and array of parameter
        values for all sample points (values).

    root : str
        Path of directory in which output files should be stored.

    class_data_root : str
        Root of filenames in which CLASS power spectra were stored.

    zvals : array_like
        Array of redshifts at which the power spectra should be evaluated.

    default_params : dict, optional
        Dictionary of default cosmological parameters, to be used if a
        necessary parameter is not included in sample_points.

    nonlin : bool, optional
        Whether to load non-linear P(k). Default: False.

    mode : str, optional
        Which mode the CLASS run was performed in. Default: 'std'.
    """

    # Get list of parameter names
    pnames = sample_points.keys()
    Nsamp = sample_points[pnames[0]].size

    # Loop over sets of parameters, getting CCL power spectra for each
    for i in range(Nsamp):
        print("Calculating CCL power spectrum %d / %d" % (i, Nsamp))

        # Load the CLASS power spectra to get k bins (lin and NL can differ!)
        class_file = "%s_%05dz1_pk.dat" % (class_data_root, i)
        k_class, pk_class = np.genfromtxt(class_file).T
        k_class *= sample_points['h'][i] # Convert to non-h^-1 units

        if nonlin:
            class_file_nl = "%s_%05dz1_pk_nl.dat" % (class_data_root, i)
            k_class_nl, pk_class_nl = np.genfromtxt(class_file_nl).T
            k_class_nl *= sample_points['h'][i] # Convert to non-h^-1 units

        # Build parameter dictionary
        params = {}
        for p in sample_points.keys():
            # Treat parameters with different naming conventions as special case
            if p == 'Omega_cdm':
                params['Omega_c'] = sample_points[p][i]
            else:
                params[p] = sample_points[p][i]

        # Instantiate CCL Cosmology object with this set of parameters
        params['transfer_function'] = 'boltzmann'
        cosmo = ccl.Cosmology(**params)

        # Loop over redshifts to get linear and nonlinear matter power spectra
        errored = []
        for j, z in enumerate(zvals):
            try:
                a = 1. / (1. + z)
                pk_lin = ccl.linear_matter_power(cosmo, k_class, a)
                if nonlin:
                    pk_nl = ccl.nonlin_matter_power(cosmo, k_class_nl, a)
            except KeyboardInterrupt:
                raise
            except:
                print "--- Error running CLASS"
                print "--- Parameters:", params
                errored.append(i)
                continue

            # Save datafiles
            if nonlin:
                fname_nl = "%s_nl_%s_%05d_z%d.dat" % (root, mode, i, j+1)
                np.savetxt(fname_nl, np.column_stack((k_class_nl, pk_nl)))
            else:
                fname_lin = "%s_lin_%s_%05d_z%d.dat" % (root, mode, i, j+1)
                np.savetxt(fname_lin, np.column_stack((k_class, pk_lin)))

    # Print runs that errored
    if len(errored) > 0: print "ERRORED:"
    for err in errored:
        print err


def generate_class_ini(sample_points, root, nonlinear=False, mnu=False,
                       redshifts=np.arange(0., 3., 0.5)):
    """
    Generate CLASS .ini files for a set of parameters.

    Parameters
    ----------
    sample_points : dict
        Dictionary containing parameter names (keys) and array of parameter
        values for all sample points (values).

    root : str
        Path of directory in which .ini files should be stored.

    nonlinear : bool, optional
        Whether CLASS should return the linear or nonlinear (halofit) power
        spectrum. Default: False.

    mnu : bool, optional
        Whether to include massive neutrinos.

    redshifts : array_like, optional
        Array of redshift values at which P(k) should be calculated. Default:
        Five bins from 0.0 to 2.5 in increments of 0.5.
    """
    # Get user-defined parameter names
    pnames = sample_points.keys()
    Nsamp = sample_points[pnames[0]].size

    # Loop over sample points
    for i in range(Nsamp):
        if i % 10 == 0: print("  Writing CLASS .ini file %d / %d" % (i, Nsamp))

        # Open file for writing
        f = open("%s_%05d.ini" % (root, i), 'w')

        # Write output location into file (will be same as .ini file location)
        f.write('root = %s_%05d\n' % (root, i))

        # Write user-defined cosmo parameters into file
        for p in pnames:
            # Handle commonly-used params that CLASS uses different names for
            if p == 'w0' or p == 'w_0':
                f.write("w0_fld = %e\n" % sample_points[p][i])
            elif p == 'wa' or p == 'w_a':
                f.write("wa_fld = %e\n" % sample_points[p][i])
            else:
                # Generic user-defined parameters
                f.write("%s = %e\n" % (p, sample_points[p][i]))

        # Write output redshifts to file
        f.write("z_pk = %s\n" % (",".join(["%3.3f" % z for z in redshifts])))

        # Write nonlinear switch to file, if specified
        if nonlinear: f.write("non linear = halofit\n")

        # Write curvature parameter
        if 'Omega_k' not in pnames: f.write('Omega_k = 0.0\n')

        # Write neutrino parameters
        N_ur = 2.0328 if mnu else 3.046
        N_ncdm = 1 if mnu else 0
        f.write("N_ur = %f\n" % N_ur)
        f.write("N_ncdm = %f\n" % N_ncdm)
        if mnu: f.write("m_ncdm = 0.06\n")

        # Add various default CLASS settings to file
        # NB: Need to set P_k_max_h/Mpc to a large value if doing halofit
        defaults = "T_cmb = 2.725\n\
#Omega_dcdmdr = 0.0\n\
#Gamma_dcdm = 0.0 \n\
#Omega_fld = 0\n\
#Omega_scf = 0\n\
#a_today = 1.\n\
#YHe = BBN\n\
#recombination = RECFAST\n\
#reio_parametrization = reio_camb\n\
#z_reio = 11.357\n\
#reionization_exponent = 1.5\n\
#reionization_width = 0.5\n\
#helium_fullreio_redshift = 3.5\n\
#helium_fullreio_width = 0.5\n\
#annihilation = 0.\n\
#annihilation_variation = 0.\n\
#annihilation_z = 1000\n\
#annihilation_zmax = 2500\n\
#annihilation_zmin = 30\n\
#annihilation_f_halo = 20\n\
#annihilation_z_halo = 8\n\
#on the spot = yes\n\
#decay = 0.\n\
output = mPk\n\
modes = s\n\
lensing = no\n\
#ic = ad\n\
#P_k_ini type = analytic_Pk\n\
#k_pivot = 0.05\n\
#alpha_s = 0.\n\
P_k_max_h/Mpc = 100.\n\
#l_max_scalars = 8000\n\
#l_max_lss = 1000\n\
headers = yes\n\
format = class\n\
write background = no\n\
write thermodynamics = no\n\
write primordial = no\n\
write parameters = yeap\n\
input_verbose = 1\n\
background_verbose = 1\n\
thermodynamics_verbose = 1\n\
perturbations_verbose = 1\n\
transfer_verbose = 1\n\
primordial_verbose = 1\n\
spectra_verbose = 1\n\
nonlinear_verbose = 1\n\
lensing_verbose = 1\n\
output_verbose = 1\n"
        f.write(defaults)
        f.close()


def run_class(fname_pattern, class_root, precision=False):
    """
    Run CLASS on a set of .ini files.

    Parameters
    ----------
    fname_pattern : str
        Pattern (glob format) for matching the .ini files to be run by CLASS.

    class_root : str
        Directory in which the 'class' executable is stored.

    precision : bool, optional
        Whether to run CLASS in high-precision mode (using the pk_ref.pre
        precision file).

    """
    # Get CLASS .ini files that will be run
    fnames = glob.glob(fname_pattern)
    fnames.sort()

    # Loop over all .ini files
    failed = []
    for i, filename in enumerate(fnames):
        print("CLASS run %d / %d (%s)" % (i+1, len(fnames), filename))

        # Run CLASS and save the output
        cmd = ['%s/class' % class_root, filename]
        if precision: cmd += ['%s/pk_ref.pre' % class_root,]
        try:
            stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            basefile = os.path.splitext('%s' % filename)[0]
            f = open('%s.txt' % basefile, 'w')
            f.write(stdout)
            f.close()
        except KeyboardInterrupt:
            raise
        except:
            raise
            # CLASS run failed
            print("    CLASS run failed. Skipping.")
            failed.append(i)


def load_summary_stats(sample_points, ccl_data_root, class_data_root,
                       thresholds=[5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                       scale_ranges = [(1e-4, 1e-2), (1e-2, 1e-1), (1e-1, 1e0)],
                       z_vals = ['1', '2', '3', '4', '5', '6'],
                       cache_name=None):
    """
    Calculate summary stats for the deviation between CCL and reference power
    spectra as a function of scale and redshift, for a large number of sample
    points over the cosmological parameter space.

    Parameters
    ----------
    sample_points : dict
        Dictionary containing parameter names (keys) and array of parameter
        values for all sample points (values).

    ccl_data_root : str
        Root of filenames of CCL power spectrum files

    class_data_root : str
        Root of filenames of CLASS power spectrum files. If the string '_nl_'
        is found in class_data_root, this will assume that nonlinear power
        spectra should be loaded.

    """
    # Get dimensions of stats array that will be constructed
    N_samp = sample_points[sample_points.keys()[0]].size
    N_thres = len(thresholds)
    N_z = len(z_vals)
    N_kbins = len(scale_ranges)

    # Check if data were cached
    if cache_name is not None:
        try:
            stats = np.load("%s.npy" % cache_name)
            print("  Loaded '%s' from cache." % cache_name)

            frac_dev = np.load("%s.frac_dev.npy" % cache_name)
            print("  Loaded '%s.frac_dev' from cache." % cache_name)

            k_arr = np.load("%s.k_arr.npy" % cache_name)
            print("  Loaded '%s.k_arr' from cache." % cache_name)

            assert stats.shape == (N_samp, N_thres, N_z, N_kbins)
            return stats, frac_dev, k_arr
        except:
            print("  Cache '%s' not found. Recomputing." % cache_name)
            pass

    # Create array to hold summary statistics, with shape:
    # (N_samp, N_thres, N_z, N_kbins)
    stats = np.zeros((N_samp, N_thres, N_z, N_kbins))
    frac_dev = [[[] for j in range(N_z)] for i in range(N_samp)]
    k_arr = [[[] for j in range(N_z)] for i in range(N_samp)]

    # Loop over sample points in parameter space and calculate summary stats
    for i in range(N_samp):
        print "  Loading power spectra for parameter set %05d" % i

        # Get Hubble parameter, h, for rescaling CLASS P(k) to Mpc units
        h = sample_points['h'][i]

        # Try to get deviation
        try:
          # Loop over redshift values
          for j in range(N_z):

            # Construct filenames for CCL and CLASS P(k) data files
            fname_ccl = "%s_%05d_z%d.dat" % (ccl_data_root, i, j+1)
            if '_nl_' in class_data_root:
                fname_class = "%s_%05dz%d_pk_nl.dat" % (class_data_root, i, j+1)
            else:
                fname_class = "%s_%05dz%d_pk.dat" % (class_data_root, i, j+1)

            # Load CCL power spectrum data
            pk_ccl_dat = np.loadtxt(fname_ccl)
            ccl_k = pk_ccl_dat[:,0]
            ccl_pk = pk_ccl_dat[:,1]

            # Load CLASS power spectrum data
            pk_class_dat = np.loadtxt(fname_class) #, skiprows=1)
            class_k = pk_class_dat[:,0]
            class_pk = pk_class_dat[:,1] / h**3.

            # Sanity checks
            print ccl_pk.size, class_pk.size
            assert ccl_pk.size == class_pk.size

            # Calculate fractional deviation
            frac_dev[i][j] = ccl_pk/class_pk - 1.
            k_arr[i][j] = ccl_k

            # Calculate summary stats in each k bin
            for m in range(N_kbins):
                kmin, kmax = scale_ranges[m]
                idxs = np.logical_and(ccl_k >= kmin, ccl_k < kmax)

                # Calculate deviation statistic, Delta, for a range of
                # threshold values (only values above the threshold are counted)
                for n, thres in enumerate(thresholds):
                    # Calculate deviation statistic
                    dev = np.log10(
                               np.abs(ccl_pk[idxs]/class_pk[idxs] - 1.) / thres)
                    dev[np.where(dev < 0.)] = 0.

                    # Store result in stats array (N_samp, N_thres, N_z, N_kbins)
                    stats[i, n, j, m] = np.sum(dev)
        except:
          raise
          # If there were any failures, set stats to 'nan' for this sample point
          print("  Failed to compute stats for sample %05d." % i)
          stats[i, :, :, :] = np.nan

    # Save to cache file
    if cache_name is not None:
        np.save(cache_name, stats)
        np.save("%s.frac_dev" % cache_name, frac_dev)
        np.save("%s.k_arr" % cache_name, k_arr)
    return stats, np.array(frac_dev), k_arr


def ccl_summary_stats(params,
                      fname_template='../stats/lhs_mpk_err_lin_%05d_z%d.dat',
                      thresholds=[5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                      scale_ranges = [(1e-4, 1e-2), (1e-2, 1e-1), (1e-1, 1e0)],
                      z_vals = ['1', '2', '3', '4', '5', '6'],
                      cache_name=None):
    """
    Calculate summary stats for the deviation between CCL and reference power
    spectra as a function of scale and redshift, for a large number of sample
    points over the cosmological parameter space.
    """
    # Get dimensions of stats array that will be constructed
    N_samp = params['id'].size
    N_thres = len(thresholds)
    N_z = len(z_vals)
    N_kbins = len(scale_ranges)

    # Check if data were cached
    if cache_name is not None:
        try:
            stats = np.load("%s.npy" % cache_name)
            print "  Loaded '%s' from cache." % cache_name
            assert stats.shape == (N_samp, N_thres, N_z, N_kbins)
            return stats, params
        except:
            raise

    # Create array to hold summary statistics, with shape:
    # (N_samp, N_thres, N_z, N_kbins)
    stats = np.zeros((N_samp, N_thres, N_z, N_kbins))

    # Loop over sample points in parameter space and calculate summary stats
    for i in range(N_samp):
        trial = params['id'][i]
        print "  Loading CCL power spectra for parameter set %05d" % i

        # Loop over redshift values
        for j in range(N_z):

            # Load cached CCL power spectrum data
            fname = fname_template % (i, z_vals[j])
            pk_ccl_dat = np.loadtxt(fname, skiprows=1)
            ccl_k = pk_ccl_dat[:,0]
            ccl_pk = pk_ccl_dat[:,1]

            # Calculate summary stats in each k bin
            for m in range(N_kbins):
                kmin, kmax = scale_ranges[m]
                idxs = np.logical_and(ccl_k >= kmin, ccl_k < kmax)

                # Calculate deviation statistic, Delta, for a range of
                # threshold values (only values above the threshold are counted)
                # FIXME: ccl_pk is actually the deviation, which was
                # precomputed somewhere!
                for n, thres in enumerate(thresholds):
                    # Calculate deviation statistic
                    dev = np.log10(np.abs(ccl_pk[idxs]) / thres)
                    dev[np.where(dev < 0.)] = 0.

                    # Store result in stats array (N_samp, N_thres, N_z, N_kbins)
                    stats[i, n, j, m] = np.sum(dev)

    # Save to cache file
    if cache_name is not None:
        np.save(cache_name, stats)

    return stats, params
