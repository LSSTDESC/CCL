# Direct number-counts reference for FKEM

The reference generator projects density and redshift-space distortions (RSD)
by direct spherical-Bessel quadrature. The saved spectra are used by
`benchmarks/test_fkem_number_counts.py` to test the public `angular_cl` API.
They cover compact radial windows, pure RSD, mixed components, cross-bin
spectra, multipoles 2 through 100, and two cosmologies.

## Definition and independence

For each compact window with endpoints `lo` and `hi`,

```text
n(z) = sin(pi*(z-lo)/(hi-lo))^4 / ((hi-lo)*3/8),  lo <= z <= hi
     = 0,                                      otherwise.

I_density(k) = integral dz n(z) b(z) D(z) j_ell(k chi(z))
I_RSD(k)     = -integral dz n(z) D(z) f(z) j_ell''(k chi(z))
C_ell       = 2/pi integral dlogk k^3 P(k, a=1) I_left(k) I_right(k).
```

The second derivative is with respect to the entire dimensionless Bessel
argument. The generator evaluates it using an order recurrence. Separate
identity tests compare the recurrence with the spherical-Bessel differential
equation and check integration by parts and redshift normalization.

Distances, growth, growth rate and the linear Eisenstein–Hu matter spectrum
are shared CCL inputs. The angular projection uses Gauss–Legendre integration
in redshift and Simpson integration in `log(k)`, independently of FKEM and
FFTLog. This checks the projection, not the background cosmology or matter
power implementation. Distances are in Mpc, wavenumbers in inverse Mpc,
power in cubic Mpc, and angular spectra are dimensionless.

## Cases and convergence

The first case uses `Omega_c=.25`, `Omega_b=.05`, `h=.7`, `sigma8=.8`,
`n_s=.96`, unit bias, and the window `[.1, 1.3]`. Its 27 comparisons use
multipoles 2, 5, and 10, `fkem_Nchi=128, 256, 512`, and
`fkem_chi_min=1`. They include density auto, density × (density + RSD),
and (density + RSD) auto spectra. All use 4097 input redshift samples.

The expanded cases use windows `[.05, .5]`, `[.9, 1.9]`, and `[.68, .82]`,
with bias `1+z` and multipoles 2, 5, 10, 30, and 100. Density, RSD, and their
sum form nine tracers, giving 45 unique pairs per multipole. Both the first
cosmology and a second with `Omega_c=.29`, `Omega_b=.05`, `h=.66`,
`sigma8=.82`, `n_s=1`, `w0=-.9`, `wa=.2` are tested, for 450 comparisons.
These cases use the default FKEM radial resolution (8194 samples for these
inputs) and default minimum distance, with `l_limber=1000` to require FKEM.
Unspecified cosmological parameters retain CCL defaults.

The reference records each quadrature configuration and the largest change
normalized by `sqrt(C_aa C_bb)` for each refinement. This normalization
remains useful for cross spectra near zero. The central reference uses 1024
radial quadrature nodes and 4097 logarithmic wavenumbers. The original case
integrates over `[1e-7, .2]` inverse Mpc, with separate lower and upper
extensions. The expanded cases integrate over `[1e-8, .4]`, with an upper
extension to `.8`. Refinements reach 2048 radial nodes and 8193 wavenumbers.
The generator requires every recorded normalized change to be below `1e-5`.
These are empirical convergence checks, not rigorous bounds on the
uncomputed infinite tail.

The benchmark tolerances are `1e-3` relative for the original coarse
128/256-point grids and `1e-4` for the original 512-point grid. All expanded
comparisons require absolute error divided by `sqrt(C_aa C_bb)` below
`1e-4`. These are regression thresholds for the specified fixtures, not a
survey accuracy guarantee. In particular, arbitrarily coarse custom radial
grids need not meet the default-resolution threshold. The existing N5K
benchmark tolerances are unchanged. The `limber_max_error` option controls
the Limber transition; it is not a total FKEM error bound.

## Range and component integration

Different Bessel derivative orders produce different FFTLog wavenumber
grids. Resampling their overlap changes the logarithmic spacing; the final
sum must use the spacing of that integration grid. Correcting this weight
alone does not control periodic-image error or missing low-wavenumber power.

FKEM therefore zero-pads the sampled radial integrand before transforming it.
Padding preserves every original radial sample and extends each end by a
decade, capped at the original sample count per side for narrow ranges.
Growth and transfer functions are evaluated only on the original radial
interval. The padding reduces periodic-image contamination and extends the
returned low-wavenumber coverage. It does not recover a physical kernel
excluded by a user-specified `fkem_chi_min`.

Each tracer-component pair is integrated over its own common wavenumber
grid, with that grid's actual `dlogk`, and the pair contributions are summed.
With unchanged radial sampling, additional derivative components therefore
do not narrow an existing pair's wavenumber overlap. The automatic Limber transition uses the same
pair contributions. The unit tests also check density/RSD additivity for
identical radial domains and repeat calculations through the transform cache.
No reference values or fitted spectral corrections enter production code.

## Reproduce

With NumPy, SciPy and a working CCL installation:

```bash
python benchmarks/data/codes/fkem_number_counts_reference.py \
  --output benchmarks/data/fkem_number_counts_reference.json
pytest benchmarks/data/codes/test_fkem_number_counts_reference.py \
  benchmarks/test_fkem_number_counts.py benchmarks/test_nonlimber.py
pytest pyccl/tests/test_cells.py pyccl/tests/test_fkem_components.py
```

Reference generation is substantially slower than running tests against the
saved JSON. The JSON records the generating dependency versions, definitions,
configurations, spectra and convergence changes. It contains independent
reference values, not outputs generated by the production projection.
