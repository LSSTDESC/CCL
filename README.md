<!---
STYLE CONVENTION USED   
    bolt italic:
        ***file***"
    code:
       `program` or `library``
       `commands` or `paths`
       `variable`
    bold code:
        **`function`**
        **`type`** or **`structure`**
-->
# CCL     
[![Build Status](https://travis-ci.org/LSSTDESC/CCL.svg?branch=master)](https://travis-ci.org/LSSTDESC/CCL) [![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CCL/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/CCL?branch=master) [![Documentation Status](https://readthedocs.org/projects/ccl/badge/?version=latest)](https://ccl.readthedocs.io/en/latest/?badge=latest)

The Core Cosmology Library (CCL) is a standardized library of routines to calculate
basic observables used in cosmology. It will be the standard analysis package used by the
LSST Dark Energy Science Collaboration (DESC).

The core functions of this package include:

  - Matter power spectra `P(k)` from numerous models including CLASS, the Mira-Titan Emulator and halofit
  - Hubble constant `H(z)` as well as comoving distances `\chi(z)` and distance moduli `\mu(z)`
  - Growth of structure `D(z)` and `f`
  - Correlation functions `C_\ell` for arbitrary combinations of tracers including galaxies, shear and number counts
  - Halo mass function `{\rm d}n/{\rm d}M` and halo bias `b(M)`
  - Approximate baryonic modifications to the matter power spectra `\Delta^2_{\rm baryons}`
  - Simple modified gravity extensions `\Delta f(z)` and `\mu-\Sigma`

This software is a publicly released LSST DESC product which was developed within the LSST
DESC using LSST DESC resources. DESC users should use it in accordance with the
[LSST DESC publication policy](http://lsstdesc.org/Collaborators). External users are
welcome to use the code outside DESC in accordance with the licensing information below.

The list of publicly released versions of this package can be found
[here](https://github.com/LSSTDESC/CCL/releases). The master branch is the most
recent (non-released) stable branch, but under development. We recommend using one
of the public releases unless working on the development on the library.

See the [documentation](https://ccl.readthedocs.io/en/latest/) for more details
and installation instructions.

## TLDR

`CCL` is available as a Python package through PyPi or conda. To install, simply run:

```
$ [pip|conda] install pyccl
```

For the PyPi installation, you will need ``CMake`` installed locally. See
[Getting CMake](https://ccl.readthedocs.io/en/latest/source/installation.html#getting-cmake)
for instructions.

Once you have the code installed, you can take it for a spin!

```python
import pyccl as ccl
import numpy as np

# Create new Cosmology object with a given set of parameters. This keeps track
# of previously-computed cosmological functions
cosmo = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks')

# Define a simple binned galaxy number density curve as a function of redshift
z_n = np.linspace(0., 1., 200)
n = np.ones(z_n.shape)

# Create objects to represent tracers of the weak lensing signal with this
# number density (with has_intrinsic_alignment=False)
lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(2, 10)
cls = ccl.angular_cl(cosmo, lens1, lens2, ell)
print(cls)
```

# License, Credits, Feedback etc

This code has been released by DESC, although it is still under active development.
It is accompanied by a journal paper that describes the development and validation of
`CCL`, which you can find on the  arxiv:[1812.05995](https://arxiv.org/abs/1812.05995).
If you make use of the ideas or software here, please cite that paper and provide a
link to this repository: https://github.com/LSSTDESC/CCL. You are welcome to re-use
the code, which is open source and available under terms consistent with our
[LICENSE](https://github.com/LSSTDESC/CCL/blob/master/LICENSE)
([BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause)).

External contributors and DESC members wishing to use CCL for non-DESC projects
should consult with the TJP working group conveners, ideally before the work has
started, but definitely before any publication or posting of the work to the arXiv.

For free use of the `CLASS` library, the `CLASS` developers require that the `CLASS`
paper be cited: CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034.
The `CLASS` repository can be found in http://class-code.net.

The `CAMB` developers have released `CAMB` under the LGPL license with a few
additional restrictions. Please read their [LICENSE](https://github.com/cmbant/CAMB/blob/master/LICENCE.txt)
for more information.

# Contact

If you have comments, questions, or feedback, please
[write us an issue](https://github.com/LSSTDESC/CCL/issues). You can also contact the
[administrators](https://github.com/LSSTDESC/CCL/CCL-administrators).
