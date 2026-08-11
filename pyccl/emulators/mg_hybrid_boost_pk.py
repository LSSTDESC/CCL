"""
mg_hybrid_boost_pk.py
======================
CCL EmulatorPk integration for the hybrid single-bin NN + multi-bin NN
MG boost emulator (Srinivasan, Prabhu, Lehman, Krishnan Vinaychandran,
Weller 2026 - arXiv:2603.11895), replacing the GP-based nonlinear
backend in mg_binned_boost_pk.py with two NN backends, routed
automatically depending on which bins are actually modified.

Bin structure
-------------------------------------------------------------------
    bin_index | MG active in z range
    ----------|----------------------
        0     |   0.0 - 0.43
        1     |   0.43 - 0.91
        2     |   0.91 - 1.47
        3     |   1.47 - 2.15
        4     |   2.15 - 3.0

Routing rule
------------
This class accepts a MODIFIED-GRAVITY PARAMETER PER BIN (mus, etas -
each length 5), rather than one (mu, eta, bin_index) triple. Given
those 5 values, it decides which nonlinear backend to use:

  - If exactly ONE bin deviates from GR (mu != 1), AND that bin is
    bin 0 or bin 4, AND every other bin is exactly at GR (mu == 1):
        -> use the single-bin NN (NonlinearBoostNN), which was only
           ever trained with one bin varying at a time, and is more
           accurate than the multi-bin NN specifically for the two
           edge bins.
  - In every other case (any middle bin (1-3) modified, more than one
    bin modified simultaneously, or all bins at GR):
        -> use the multi-bin NN (the joint 5-mu model), since it is
           the only backend that ever saw multiple bins varying
           together during training and can capture the resulting
           cross-bin interaction terms.

IMPORTANT CAVEAT: this routing does NOT recover cross-bin interaction
terms between {bin 0, bin 4} and the other bins in cases where BOTH an
edge bin and another bin are modified simultaneously - that case
falls to the multi-bin model, which handles it correctly on its own
(no combination needed in that branch). The single-bin path is only
ever taken when it's provably equivalent to what the model was
trained on (exactly one bin, that bin is an edge bin, everything else
at GR) - so no cross-bin interaction is ever silently dropped by
this router. See discussion with C. Anthropic, July 2026, on why a
post-hoc combination of separately-computed single-bin and multi-bin
contributions is NOT used here.

eta asymmetry -- unchanged from the GP version
------------------------------------------------
eta is a parameter of the linear-NN component only (linear scales,
k < ~0.01 h/Mpc). Neither the single-bin nonlinear NN nor the
multi-bin nonlinear NN use eta directly for the nonlinear boost
itself; both stitching layers call their linear and nonlinear
components independently and blend them, so passing eta through is
still correct and complete.

Units
------
  emulator k  ->  h/Mpc
  CCL Pk2D k  ->  1/Mpc   (conversion: k_ccl = k_emu * h, handled here)
"""

__all__ = ("MGHybridBoostPk",)

import os
import warnings
import numpy as np
from pathlib import Path
from .. import Pk2D
from . import EmulatorPk

# -- Bin definitions ---------------------------------------------------------
BIN_Z_RANGES = {
    0: (0.0, 0.43),
    1: (0.43, 0.91),
    2: (0.91, 1.47),
    3: (1.47, 2.15),
    4: (2.15, 3.0),
}

EDGE_BINS = (0, 4)
N_BINS = 5

# -- Training bounds (from emulator/utils.py -- same for both NN backends) --
_A_S_MIN = np.exp(2.9960) / 1e10
_A_S_MAX = np.exp(3.0910) / 1e10

_COSMO_BOUNDS = {
    "Omega_m": (0.25, 0.35),
    "Omega_b": (0.04, 0.055),
    "h": (0.65, 0.73),
    "n_s": (0.95, 1.0),
    "A_s": (_A_S_MIN, _A_S_MAX),
}
_MU_BOUNDS = (0.9, 1.1)
_ETA_BOUNDS = (0.9, 1.1)
_Z_BOUNDS = (0.0, 3.0)

# tolerance for treating a mu value as "exactly GR" (mu == 1.0)
_GR_ATOL = 1e-8


class MGHybridBoostPk(EmulatorPk):
    """Non-linear P(k,z) for binned phenomenological modified gravity,
    using a hybrid single-bin NN / multi-bin NN backend.

    Computed as:

        P_MG(k, z) = P_LCDM(k, z)  x  B(k, z; mus, etas)

    where B is the MG boost, computed by whichever backend the
    routing rule (see module docstring) selects.

    Parameters
    ----------
    mus : array-like of length 5, optional
        MG parameter mu per bin (0-4). Defaults to all 1.0 (pure GR,
        boost == 1 everywhere). Each entry should lie in [0.9, 1.1].
    etas : array-like of length 5, optional
        MG parameter eta per bin (0-4). Defaults to all 1.0. Each
        entry should lie in [0.9, 1.1]. Used by the linear-NN
        component only.
    model_dir_single : str, optional
        Directory containing the single-bin NN model files
        (linear_boost_nn.pt, singlebin_nn_emulator.pt, cola_eg.txt).
        Defaults to ``<pyccl>/emulators/data/mg_singlebin_nn/``.
    model_dir_multi : str, optional
        Directory containing the multi-bin NN model files
        (linear_boost_nn_multibin.pt, multibin_nn_emulator.pt).
        Defaults to ``<pyccl>/emulators/data/mg_multibin_nn/``.
    baseline_pk : str or EmulatorPk, optional
        CCL nonlinear method for the LCDM baseline. Defaults to
        ``'halofit'``.
    z_arr : array-like, optional
        Redshift grid for boost evaluation. Must lie in [0.0, 3.0].
        Defaults to 30 points log-spaced from 0.01 to 3.0 - always
        spans the full range so growth history from all bins is
        captured regardless of which bins are modified.
    k_arr : array-like, optional
        Wavenumber grid in h/Mpc. Defaults to 200 log-spaced points
        from 1e-3 to 1.0 h/Mpc. Do not extend beyond 1.0 h/Mpc -
        both NN backends show growing disagreement with their GP/
        training targets above k~1 h/Mpc (COLA accuracy limitation,
        same caveat as the GP-based integration).

    Notes
    -----
    * ``A_s`` must be provided in the CCL Cosmology (not ``sigma8``).
    * The emulator returns k in h/Mpc; CCL's Pk2D expects 1/Mpc.
      Conversion is handled internally.
    * Unlike the GP-based MGBinnedBoostPk, this class accepts all 5
      bins' mu/eta values at once and figures out internally which
      backend to use - you do not need 5 separate Cosmology objects
      unless you specifically want 5 independent single-bin-only
      results.
    * Requires the ``single_bin_emulator`` and ``multi_bin_emulator``
      source folders to be present as sub-packages directly inside
      ``pyccl/emulators/``,which the packaging already addresses.
    """

    def __init__(
        self,
        mus=None,
        etas=None,
        model_dir_single=None,
        model_dir_multi=None,
        baseline_pk='halofit',
        z_arr=None,
        k_arr=None,
    ):
        self.mus = (np.ones(N_BINS) if mus is None
                    else np.asarray(mus, dtype=float))
        self.etas = (np.ones(N_BINS) if etas is None
                     else np.asarray(etas, dtype=float))
        self.baseline_pk = baseline_pk

        if len(self.mus) != N_BINS or len(self.etas) != N_BINS:
            raise ValueError(
                f"MGHybridBoostPk: mus and etas must each have "
                f"{N_BINS} entries (one per bin), got "
                f"{len(self.mus)} and {len(self.etas)}."
            )

        self.z_arr = (np.logspace(np.log10(0.01), np.log10(3.0), 30)
                      if z_arr is None else np.asarray(z_arr, dtype=float))

        self.k_arr = (np.logspace(-3, 0, 200)
                      if k_arr is None else np.asarray(k_arr, dtype=float))

        if model_dir_single is None:
            model_dir_single = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data', 'mg_singlebin_nn'
            )
        if model_dir_multi is None:
            model_dir_multi = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data', 'mg_multibin_nn'
            )
        self.model_dir_single = model_dir_single
        self.model_dir_multi = model_dir_multi

        self._validate_mg_params()
        self._load_emulators()

    # -- Validation -----------------------------------------------------

    def _validate_mg_params(self):
        for i in range(N_BINS):
            for name, val, bounds in [
                (f"mus[{i}]", self.mus[i], _MU_BOUNDS),
                (f"etas[{i}]", self.etas[i], _ETA_BOUNDS),
            ]:
                lo, hi = bounds
                if not (lo <= val <= hi):
                    warnings.warn(
                        f"MGHybridBoostPk: {name}={val} outside training "
                        f"range [{lo}, {hi}]. Accuracy not guaranteed.",
                        UserWarning
                    )

    def _validate_cosmo(self, cosmo_dict):
        for name, val in cosmo_dict.items():
            if name in _COSMO_BOUNDS:
                lo, hi = _COSMO_BOUNDS[name]
                if not (lo <= val <= hi):
                    warnings.warn(
                        f"MGHybridBoostPk: {name}={val:.4e} outside "
                        f"training range [{lo:.4e}, {hi:.4e}].",
                        UserWarning
                    )

    def _validate_z_arr(self):
        z_lo, z_hi = _Z_BOUNDS
        if np.any(self.z_arr < z_lo) or np.any(self.z_arr > z_hi):
            warnings.warn(
                f"MGHybridBoostPk: z_arr contains values outside the "
                f"training range [{z_lo}, {z_hi}].",
                UserWarning
            )

    # -- Loading ----------------------------------------------------------

    def _load_emulators(self):
        try:
            from .single_bin_emulator import MGEmulator as SingleBinMGEmulator
        except ImportError:
            raise ImportError(
                "Could not import 'pyccl.emulators.single_bin_emulator."
                "MGEmulator'. Make sure the single_bin_emulator source "
                "folder has been copied into pyccl/emulators/ (as a "
                "sub-package, alongside this file), and set "
                "model_dir_single to the directory containing its "
                "model files (linear_boost_nn.pt, "
                "singlebin_nn_emulator.pt, bin5_nn_emulator.pt, "
                "cola_eg.txt)."
            )
        try:
            from .multi_bin_emulator import (
                LinearEmulator as MultiBinLinearEmulator,
                NonLinearEmulator as MultiBinNonLinearEmulator,
                MGEmulator as MultiBinMGEmulator,
            )
        except ImportError:
            raise ImportError(
                "Could not import from 'pyccl.emulators."
                "multi_bin_emulator'. Make sure the multi_bin_emulator "
                "source folder has been copied into pyccl/emulators/ "
                "(as a sub-package, alongside this file), and set "
                "model_dir_multi to the directory containing its "
                "model files (linear_boost_nn_multibin.pt, "
                "multibin_nn_emulator.pt)."
            )

        self._single_emu = SingleBinMGEmulator(model_dir=self.model_dir_single)

        # -----------------------------------------------------
        # Unlike the single-bin MGEmulator (which builds
        # everything internally from a model_dir), the multi-bin
        # MGEmulator expects pre-built LinearEmulator/
        # NonLinearEmulator OBJECTS, matching the pattern in the
        # package's own reference emulator.py script.
        #
        # k_native here is NOT stored in the checkpoint (unlike
        # the single-bin package's cola_eg.txt) - it's supplied
        # externally, exactly as in the reference script. This
        # grid is present as "k.npy" file( already present
        # in multi_bin emulator directory).
        # -----------------------------------------------------
        multi_linear = MultiBinLinearEmulator(
            self.model_dir_multi + "/linear_boost_nn_multibin.pt"
        )
        k_path = Path(__file__).resolve().parent / \
            "multi_bin_emulator" / "k.npy"
        k_nl = np.load(k_path)

        multi_nonlinear = MultiBinNonLinearEmulator(
            self.model_dir_multi + "/multibin_nn_emulator.pt",
            k_nl,
        )

        self._multi_emu = MultiBinMGEmulator(multi_linear, multi_nonlinear)

    # -- Cosmology dict -----------------------------------------------------

    def _ccl_cosmo_to_dict(self, cosmo):
        """Build the dict expected by the single-bin emulator's
        predict_boost, and also used to build the multi-bin call.

        Keys: Omega_m, Omega_b, h, n_s, A_s (raw A_s, not lnAs).
        """
        A_s = cosmo['A_s']
        if np.isnan(A_s):
            raise ValueError(
                "MGHybridBoostPk requires A_s in the CCL Cosmology "
                "(not sigma8). The emulators train on ln(10^10 A_s) "
                "and cannot derive A_s from sigma8 without an "
                "iterative solve."
            )
        cosmo_dict = {
            'Omega_m': float(cosmo['Omega_m']),
            'Omega_b': float(cosmo['Omega_b']),
            'h': float(cosmo['h']),
            'n_s': float(cosmo['n_s']),
            'A_s': float(A_s),
        }
        self._validate_cosmo(cosmo_dict)
        return cosmo_dict

    # -- Routing --------------------------------------------------------

    def _modified_bins(self):
        return [
            i for i in range(N_BINS)
            if not np.isclose(self.mus[i], 1.0, atol=_GR_ATOL)
        ]

    def _use_single_bin(self):
        modified = self._modified_bins()
        return len(modified) == 1 and modified[0] in EDGE_BINS

    # -- Boost grid -----------------------------------------------------

    def _compute_boost_grid(self, cosmo_dict):
        """Evaluate the MG boost on (z_arr, k_arr), routing to the
        single-bin or multi-bin NN backend depending on which bins
        are modified.

        Returns
        -------
        boost : ndarray, shape (n_z, n_k)
            B(z, k) = P_MG / P_LCDM, k in h/Mpc.
        """
        if self._use_single_bin():
            bin_index = self._modified_bins()[0]
            mu = float(self.mus[bin_index])
            eta = float(self.etas[bin_index])

            k_emu, boost_raw = self._single_emu.predict_boost(
                cosmo_dict,
                mu=mu,
                eta=eta,
                bin_index=bin_index,
                zs=self.z_arr,
            )
            self._last_route = f"single-bin (bin_index={bin_index})"

        else:
            k_emu, boost_raw = self._multi_emu.predict(
                zs=self.z_arr,
                omega_m=cosmo_dict["Omega_m"],
                omega_b=cosmo_dict["Omega_b"],
                h=cosmo_dict["h"],
                n_s=cosmo_dict["n_s"],
                A_s=np.log(1e10*cosmo_dict["A_s"]),
                mus=self.mus,
                etas=self.etas,
                k=None,   # keep native grid here; interpolate below
                          # onto self.k_arr the same way for both branches
            )
            self._last_route = (
                f"multi-bin (modified bins={self._modified_bins()})"
            )

        k_emu = np.asarray(k_emu)
        boost_raw = np.asarray(boost_raw)

        # Interpolate onto self.k_arr (same units: h/Mpc),
        boost_grid = np.ones((len(self.z_arr), len(self.k_arr)))
        for iz in range(len(self.z_arr)):
            boost_grid[iz] = np.interp(
                self.k_arr,
                k_emu,
                boost_raw[iz],
                left=boost_raw[iz, 0],
                right=1.0,
            )
        return boost_grid

    # -- Linear-only boost (before blending with nonlinear) ---------------

    def _compute_linear_boost_grid(self, cosmo_dict):
        """Evaluate the LINEAR-only MG boost (the linear-NN component
        alone, BEFORE it gets blended with the nonlinear component
        inside stitching.py's BoostEmulator/MGEmulator), on
        (z_arr, k_arr). Uses the same single-bin/multi-bin routing
        rule as _compute_boost_grid.

        Returns
        -------
        boost_lin : ndarray, shape (n_z, n_k)
        """
        if self._use_single_bin():
            bin_index = self._modified_bins()[0]
            mu = float(self.mus[bin_index])
            eta = float(self.etas[bin_index])

            # self._single_emu.emulator is the BoostEmulator built in
            # single_bin_emulator/mg_emulator.py; .linear is the
            # LinearBoostNN instance - calling it directly bypasses
            # the nonlinear blend entirely.
            k_emu, boost_raw = self._single_emu.emulator.linear.predict_boost(
                cosmo_dict, mu=mu, eta=eta, bin_index=bin_index, zs=self.z_arr,
            )
            k_emu = np.asarray(k_emu)
            boost_raw = np.asarray(boost_raw)

            boost_grid = np.ones((len(self.z_arr), len(self.k_arr)))
            for iz in range(len(self.z_arr)):
                boost_grid[iz] = np.interp(
                    self.k_arr,
                    k_emu,
                    boost_raw[iz],
                    left=boost_raw[iz, 0],
                    right=1.0,
                )

        else:
            # self._multi_emu.linear is the LinearEmulator instance
            # built in _load_emulators - calling it directly bypasses
            # the nonlinear blend. It expects ln(1e10*A_s), same as
            # the nonlinear branch's fix above, and takes one z at a
            # time (unlike the nonlinear/full MGEmulator.predict,
            # which accepts an array of zs).
            ln1e10As = np.log(1e10 * cosmo_dict["A_s"])

            boost_grid = np.empty((len(self.z_arr), len(self.k_arr)))
            for iz, z in enumerate(self.z_arr):
                boost_grid[iz] = self._multi_emu.linear.predict(
                    omega_m=cosmo_dict["Omega_m"],
                    omega_b=cosmo_dict["Omega_b"],
                    h=cosmo_dict["h"],
                    n_s=cosmo_dict["n_s"],
                    ln1e10As=ln1e10As,
                    mus=self.mus,
                    etas=self.etas,
                    z_pk=z,
                    k=self.k_arr,
                )

        return boost_grid

    # -- CCL interface ----------------------------------------------------

    def _get_pk2d(self, cosmo):
        """Build Pk2D for P_MG = P_LCDM x Boost.

        Called by CCL whenever nonlinear_matter_power is requested.
        """
        import pyccl

        self._validate_z_arr()
        cosmo_dict = self._ccl_cosmo_to_dict(cosmo)
        h = float(cosmo['h'])

        # Step 1 -- boost on (z_arr, k_arr), k in h/Mpc
        boost = self._compute_boost_grid(cosmo_dict)   # (n_z, n_k)

        # Step 2 -- LCDM baseline P(k, z) from CCL
        k_ccl = self.k_arr * h   # 1/Mpc (handling unit conversion internally)
        a_arr = 1.0 / (1.0 + self.z_arr)

        pk_lcdm = np.array([
            [pyccl.nonlin_matter_power(cosmo, ki, ai) for ki in k_ccl]
            for ai in a_arr
        ])

        # Step 3 -- multiply
        pk_mg = pk_lcdm * boost   # (n_z, n_k)

        # Step 4 -- pack into Pk2D (requires ascending a)
        a_asc = a_arr[::-1]
        pk_asc = pk_mg[::-1]

        return Pk2D(
            a_arr=a_asc,
            lk_arr=np.log(k_ccl),
            pk_arr=np.log(pk_asc),
            is_logp=True,
            extrap_order_lok=1,
            extrap_order_hik=2,
        )

    def get_linear_pk(self, cosmo):
        """Return Pk2D for the LINEAR-only MG power spectrum:

            P_MG_linear(k,z) = P_LCDM_linear(k,z)  x  B_linear(k,z)

        where B_linear is the linear-NN boost component ALONE -
        i.e. the boost as computed in stitching.py BEFORE it gets
        combined with the nonlinear component to make the full
        hybrid boost. Uses pyccl.linear_matter_power as the LCDM
        baseline (the linear power spectrum).
        """
        import pyccl

        self._validate_z_arr()
        cosmo_dict = self._ccl_cosmo_to_dict(cosmo)
        h = float(cosmo['h'])

        boost_lin = self._compute_linear_boost_grid(cosmo_dict)   # (n_z, n_k)

        k_ccl = self.k_arr * h   # 1/Mpc ( conversion handling)
        a_arr = 1.0 / (1.0 + self.z_arr)

        pk_lcdm_lin = np.array([
            [pyccl.linear_matter_power(cosmo, ki, ai) for ki in k_ccl]
            for ai in a_arr
        ])

        pk_mg_lin = pk_lcdm_lin * boost_lin

        a_asc = a_arr[::-1]
        pk_asc = pk_mg_lin[::-1]

        return Pk2D(
            a_arr=a_asc,
            lk_arr=np.log(k_ccl),
            pk_arr=np.log(pk_asc),
            is_logp=True,
            extrap_order_lok=1,
            extrap_order_hik=2,
        )

    # --Returns boost grid for user ------------------------------------------

    def boost(self, cosmo):
        """Return the full linear+nonlinear MG boost.

    B(k,z) = P_MG(k,z) / P_LCDM(k,z)

    Returns
    -------
    k_arr : ndarray
        Wavenumber grid in 1/Mpc.
    z_arr : ndarray
        Redshift grid.
    boost_grid : ndarray
        Boost grid with shape (n_z, n_k).
        """

        self._validate_z_arr()
        cosmo_dict = self._ccl_cosmo_to_dict(cosmo)

        boost = self._compute_boost_grid(cosmo_dict)

        k_arr = self.k_arr * cosmo["h"]

        return k_arr, self.z_arr, boost

    # -- Convenience ------------------------------------------------------

    @property
    def modified_bins(self):
        """List of bin indices currently deviating from GR."""
        return self._modified_bins()

    @property
    def route(self):
        """Which backend was used on the last _compute_boost_grid call.
        None until _get_pk2d has been called at least once.
        """
        return getattr(self, "_last_route", None)

    def __repr__(self):
        return (
            f"MGHybridBoostPk("
            f"mus={list(self.mus)}, etas={list(self.etas)}, "
            f"modified_bins={self.modified_bins}, "
            f"baseline='{self.baseline_pk}')"
        )
