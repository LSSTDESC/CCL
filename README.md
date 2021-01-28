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
$ conda install -c conda-forge pyccl
```

or

```
$ pip install pyccl
```

For the PyPi installation, you will need ``CMake`` installed locally. See
[Getting CMake](https://ccl.readthedocs.io/en/latest/source/installation.html#getting-cmake)
for instructions. Note that the code only supports Linux or Mac OS, but no Windows.

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
[write us an issue](https://github.com/LSSTDESC/CCL/issues). 

The current lead of the LSST DESC CCL Topical Team is Danielle Leonard (c-d-leonard, danielle.leonard at ncl.ac.uk)


# Acknowledgements

The DESC acknowledges ongoing support from the Institut National de Physique Nucleaire et de Physique des Particules in France; the Science \& Technology Facilities Council in the United Kingdom; and the Department of Energy, the National Science Foundation, and the LSST Corporation in the United States.  DESC uses resources of the IN2P3 Computing Center (CC-IN2P3--Lyon/Villeurbanne - France) funded by the Centre National de la Recherche Scientifique; the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231; STFC DiRAC HPC Facilities, funded by UK BIS National E-infrastructure capital grants; and the UK particle physics grid, supported by the GridPP Collaboration.  This work was performed in part under DOE Contract DE-AC02-76SF00515.

NEC acknowledges support from a Royal Astronomical Society research fellowship and the Delta ITP consortium, a program of the Netherlands Organisation for Scientific Research (NWO) that is funded by the Dutch Ministry of Education, Culture and Science (OCW). DA acknowledges support from the Science and Technology Facilities Council through an Ernest Rutherford Fellowship, grant reference ST/P004474. AL and CG acknowledge support from the European Research Council under the European Union's Seventh Framework Programme (FP/2007-2013) / ERC Grant Agreement No. [616170] for work on the generic interface for theory inputs. 
