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
# CCL     [![Build Status](https://travis-ci.org/LSSTDESC/CCL.svg?branch=master)](https://travis-ci.org/LSSTDESC/CCL) [![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CCL/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/CCL?branch=master)
LSST DESC Core Cosmology Library (`CCL`) provides routines to compute basic cosmological observables with validated numerical accuracy. The library is written in C99 and all functionality is directly callable from C and C++ code.  We also provide python bindings for higher-level functions.

This software is a publicly released LSST DESC product which was developed within the LSST DESC using LSST DESC resources. DESC users should use it in accordance with the [LSST DESC publication policy](http://lsstdesc.org/Collaborators). External users are welcome to use the code outside DESC in accordance with the licensing information below.

The list of publicly released versions of this package can be found [here](https://github.com/LSSTDESC/CCL/releases). The master branch is the most recent (non-released) stable branch, but under development. We recommend using one of the public releases unless working on the development on the library.

Installation instructions can be found in [INSTALL.md](https://github.com/LSSTDESC/CCL/blob/master/INSTALL.md) in this directory. Documentation for `CCL` can be found:
* in [DOC.md](https://github.com/LSSTDESC/CCL/blob/master/DOC.md) in this directory for an overview, 
* in our [wiki](https://github.com/LSSTDESC/CCL/wiki) for a description of benchmarks and known installation issues, 
* in the `CCL` [readthedocs page](https://readthedocs.org/projects/ccl/) for the `python` routines,  
* by calling `help(function name)` from within `python`, and
* in the doxygen docs contained in the `doc` folder within the repository for the C routines. 
* There are also multiple examples in C, python and jupyter notebooks available in the `examples` folder.


For known installation issues and further information on how CCL was benchmarked during development, see our [wiki](https://github.com/LSSTDESC/CCL/wiki).

# License, Credits, Feedback etc
This code has been released by DESC, although it is still under active development. When it reaches v1 status it will be accompanied by a journal paper that describes the development and validation of `CCL`. You are welcome to re-use the code, which is open source and available under terms consistent with our [LICENSE](https://github.com/LSSTDESC/CCL/blob/master/LICENSE), which is a [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license. If you make use of any of the ideas or software in this package in your own research, please cite them as "(LSST DESC, in preparation)" and provide a link to this repository: https://github.com/LSSTDESC/CCL. For free use of the `CLASS` library, the `CLASS` developers require that the `CLASS` paper be cited: CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034. The `CLASS` repository can be found in http://class-code.net. Finally, CCL uses code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We have obtained permission from the FFTLog author to include modified versions of his source code.

# Contact
If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/CCL/issues). You can also contact the [administrators](https://github.com/LSSTDESC/CCL/CCL-administrators).
