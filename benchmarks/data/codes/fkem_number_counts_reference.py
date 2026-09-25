"""Generate direct density/RSD angular-projection reference spectra.

Radial Gauss-Legendre quadrature and outer Simpson integration are independent
of FFTLog. Distances, growth and linear matter power are shared CCL inputs.
"""
import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path

import numpy as np
import pyccl as ccl
import scipy
from scipy.integrate import simpson
from scipy.special import roots_legendre, spherical_jn

ZLO, ZHI = 0.1, 1.3
COSMOLOGIES = [
    dict(Omega_c=.25, Omega_b=.05, h=.7, sigma8=.8, n_s=.96),
    dict(Omega_c=.29, Omega_b=.05, h=.66, sigma8=.82, n_s=1.,
         w0=-.9, wa=.2),
]
WINDOWS = [('low', .05, .5), ('high', .9, 1.9), ('narrow', .68, .82)]
ELLS = [2, 5, 10, 30, 100]


def redshift_distribution(z, lo=ZLO, hi=ZHI):
    """Unit-normalized compact sin^4 distribution."""
    z = np.asarray(z)
    value = np.sin(np.pi * (z-lo)/(hi-lo))**4 / ((hi-lo)*3/8)
    return np.where((z >= lo) & (z <= hi), value, 0.)


def bessel_second_derivative(ell, x):
    """Differentiate the whole dimensionless argument using the Bessel ODE."""
    return ((ell*(ell+1)/x**2 - 1)*spherical_jn(ell, x)
            - 2/x*spherical_jn(ell, x, derivative=True))


class Reference:
    """Direct density, RSD and total projections for compact radial windows."""

    def __init__(self, cosmo, windows, bias_power):
        self.cosmo = cosmo
        self.windows = windows
        self.bias_power = bias_power

    @lru_cache(maxsize=24)
    def radial_rule(self, index, order):
        _, lo, hi = self.windows[index]
        nodes, weights = roots_legendre(order)
        z = lo + (nodes+1)*(hi-lo)/2
        a = 1/(1+z)
        chi = ccl.comoving_radial_distance(self.cosmo, a)
        w = (weights*(hi-lo)/2*redshift_distribution(z, lo, hi)
             * ccl.growth_factor(self.cosmo, a))
        return chi, w*(1+z)**self.bias_power, w*ccl.growth_rate(self.cosmo, a)

    def project(self, ell, k, radial_order):
        output = []
        for index in range(len(self.windows)):
            chi, density_w, rsd_w = self.radial_rule(index, radial_order)
            density, rsd = np.empty((2, len(k)))
            for start in range(0, len(k), 128):
                end = min(len(k), start+128)
                x = k[start:end, None]*chi[None, :]
                jl = spherical_jn(ell, x)
                # An order recurrence provides a separate expression from
                # the ODE derivative used in the reference identity tests.
                jsecond = (
                    ell*(ell-1)/((2*ell+1)*(2*ell-1))
                    * spherical_jn(ell-2, x)
                    - (ell**2/(2*ell-1)+(ell+1)**2/(2*ell+3))/(2*ell+1)*jl
                    + (ell+1)*(ell+2)/((2*ell+1)*(2*ell+3))
                    * spherical_jn(ell+2, x))
                density[start:end] = jl @ density_w
                rsd[start:end] = -jsecond @ rsd_w
            output.extend([density, rsd, density+rsd])
        return np.array(output)

    def integrate(self, ell, radial_order, k_samples, kmin, kmax):
        k = np.geomspace(kmin, kmax, k_samples)
        p = self.project(ell, k, radial_order)
        weight = 2/np.pi*k**3*ccl.linear_matter_power(self.cosmo, k, 1.)
        return simpson(p[:, None, :]*p[None, :, :]*weight,
                       x=np.log(k), axis=-1)


def generate(cosmology, windows, bias_power, ells, configurations):
    cosmo = ccl.Cosmology(**cosmology, transfer_function='eisenstein_hu',
                          matter_power_spectrum='linear')
    reference = Reference(cosmo, windows, bias_power)
    pairs = list(itertools.combinations_with_replacement(
        range(3*len(windows)), 2))
    result = dict(cosmology=cosmology, windows=windows, bias_power=bias_power,
                  pairs=pairs, configurations=configurations, spectra=[])
    for ell in ells:
        values = {}
        for label, nr, nk, lo, hi in configurations:
            values[label] = reference.integrate(ell, nr, nk, lo, hi)
            print(windows[0][0], cosmology['h'], ell, label, flush=True)
        central = values['outer']
        norm = np.sqrt(np.diag(central)[:, None]*np.diag(central)[None, :])
        deltas = {label: float(np.max(abs(v-central)/norm))
                  for label, v in values.items()}
        if max(deltas.values()) >= 1.e-5:
            raise RuntimeError(f'Reference refinement failed: {ell}: {deltas}')
        result['spectra'].append(dict(
            ell=ell, values=[float(central[i, j]) for i, j in pairs],
            normalized_refinement_changes=deltas))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        versions=dict(pyccl=ccl.__version__, numpy=np.__version__,
                      scipy=scipy.__version__),
        definition='2/pi integral dlogk k^3 P(k,1) I_a(k) I_b(k)',
        components=['density', 'RSD', 'density+RSD'],
        rsd_sign='-integral dz n(z) D(z) f(z) j_ell_second(k chi(z))',
        units=dict(chi='Mpc', k='Mpc^-1', P='Mpc^3', Cl='dimensionless'),
        shared_inputs='CCL background, growth and linear power',
        limitation='Finite-domain refinements are empirical tail checks.',
        cases=[])
    original = [('base', 512, 2049, 1.e-7, .2),
                ('radial', 1024, 2049, 1.e-7, .2),
                ('outer', 1024, 4097, 1.e-7, .2),
                ('lower', 1024, 4097, 1.e-8, .2),
                ('upper', 2048, 8193, 1.e-7, .4)]
    report['cases'].append(generate(
        COSMOLOGIES[0], [('original', ZLO, ZHI)], 0, [2, 5, 10], original))
    expanded = [('base', 512, 2049, 1.e-8, .4),
                ('radial', 1024, 2049, 1.e-8, .4),
                ('outer', 1024, 4097, 1.e-8, .4),
                ('upper', 2048, 8193, 1.e-8, .8)]
    for cosmology in COSMOLOGIES:
        report['cases'].append(generate(
            cosmology, WINDOWS, 1, ELLS, expanded))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
