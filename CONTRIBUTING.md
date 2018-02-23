
HOW TO CONTRIBUTE
---------------------------------------

This file contains guidelines on how to contribute to the development of CCL. 
If you want to make changes to CCL, please follow the procedures listed here 
to make sure that they conform to our coding standards, and that all of the 
necessary files, wrappers etc. are properly updated.

Reviewing a pull request (PR) on github
---------------------------------------
 1. Checkout the branch.
 2. Make sure you can install the C library.
 3. Make sure you can install the python module (and have regenerated the 
    SWIG wrapper if necessary; see below).
 4. Make sure the C unit tests pass (i.e. run `make check` successfully).
 5. Make sure the python tests pass (i.e. run `python run_tests.py` from the 
    `tests/` directory).
 6. Make sure that the autotools files and SWIG wrapper have been regenerated 
    if necessary.
 7. Look at the code (see "Files changed" on the top of the GitHub pull request 
    page) and check that the changes make sense to you.
 8. If new science has been implemented, and if possible, try to compare the 
    output of the code against your own predictions. Ask the developer to 
    implement appropriate unit tests for it.
 9. Make sure that the unit tests pass on Travis-CI.
10. Make sure that the changes come with documentation, e.g. internally in the 
    C code and through Python docstrings, and that the doxygen documentation 
    has been regenerated. Make sure that example code in the `examples/` 
    directory has been updated appropriately, and that the CCL note has been 
    updated if necessary.

Things to do if you are adding new features to the CCL C lib
------------------------------------------------------------
 1. If you haven't created a new files (i.e. you only edited an existing one), 
    you don't need to do any of this.
 2. When adding a new source file (.c), put it in `src/`. The new file should 
    be listed under `libccl_la_SOURCES` in `Makefile.am`.
 3. When adding a new header file (.h), put it in `include/`. The new file 
    should be listed under `include_HEADERS` in `include/Makefile.am`. It 
    should also have `doxygen`-compatible documentation strings.
 4. When adding new unit test files, they should be listed under 
    `check_ccl_SOURCES` in `Makefile.am`.
 5. Any other new files that should be included with the library (e.g. 
    documentation files) should be listed under `EXTRA_DIST` in `Makefile.am`.
 6. The file `configure.ac` may have to be modified when adding new features 
    (e.g. adding a new dependency).
 7. After making changes to any of these files, developers should run `autoreconf -i`.
    and commit any modified files (some files will be automatically regenerated 
    by this command).
 8. Make sure that you regenerate the SWIG Python wrapper (see below).
 9. Any updates to CCL should be documented in the CCL note that you can find 
    in `doc/0000-ccl_note/`. You will need to edit `main.tex`, compile with `make`,
    and commit your changes, including the updated PDF file.
10. If your changes break the CCL API, observe the guidelines in the relevant 
    section below.

Notes on autotools:
 - To install the library, users will run:
     ```sh
     $ configure <options>
     $ make
     $ make install
     ```
 - Once the library is compiled, all unit tests are run by typing `make check`.
 - A tarball with all files needed to install the library can be created 
   automatically by typing `make dist`.

To view the doxygen documentation, open any .html file in the `html/` 
directory. To refresh the docs to reflect new changes, run 
`doxygen doxygen/Doxyfile` in the directory `doc` (assuming you already have 
it installed).

Modifying the Python wrapper
---------------------------------------------------------
If you make changes to the public CCL API, you will also have to update the 
Python wrapper. The most important task is to make sure your new or modified 
functions are properly exposed to the user through the wrapper. This should be 
quite straightforward in most instances.

To do this, examine the following files (where <modulename> is the name of the 
CCL module that your new/modified function belongs to; e.g. 'background', if 
the function is in `ccl_background.c`):

 - `pyccl/<modulename>.i`: These are the SWIG interface files, and mostly handle 
   things like vectorizing C functions. This is usually done by adding a loop 
   over the indices of the input array, and by providing a function call 
   signature that can be used to accept numpy arrays. See function names ending 
   in `_vec` in the interface file for examples. If you changed the call 
   signature of a function, there is a good chance you'll have to also change 
   it in the corresponding interface file.
 - `pyccl/<modulename>.py`: These are Python wrappers around the auto-generated 
   SWIG wrapper (which is only exposed through the `pyccl.lib` namespace). They 
   are there to present a more 'Pythonic' interface to the user, for example by 
   wrapping CCL structs in more user-friendly classes, automatically handling 
   memory management logic, and doing type checking and error checking.
   These files are where you should define classes to manage CCL objects (see 
   the `Parameters` and `Cosmology` classes in `pyccl/core.py` for example), and 
   where you should provide easy-to-use wrappers around more complicated 
   functions provided through the basic SWIG wrapper. Ideally, you would also 
   provide some type checking and error checking code in this part of the 
   wrapper.
 - `pyccl/__init__.py`: This file is used to define the top-level user 
   interface of the `pyccl` module. You should make sure that your new/modified 
   functions are imported.

If you added new functions to the interface, you should also take a look in 
the following files:

 - `pyccl/ccl.i`: This is the root interface file, and includes all of the 
   interface files. If you added a new module to CCL, you will probably need to 
   create a `<modulename>.i` file for it. Once you've done that, include it in 
   `ccl.i` and it should be automatically picked up by the build system.
 - `pyccl/pyutils.py`: This file contains convenience functions for calling CCL 
   functions that take numpy array input/output arguments, and for passing the 
   `Cosmology` class as an argument to a function in the SWIG wrapper. Use 
   these if possible, instead of writing your own wrapper code for individual 
   functions.

After these changes have been made, you should regenerate `pyccl/ccl_wrap.c` and
`pyccl/ccllib.py` by running `swig`:
  ```sh
  $ swig -python -threads -o pyccl/ccl_wrap.c pyccl/ccl.i
  ```
This command may fail if swig can't find the GSL headers, in which case
add `-I/path/to/gsl/headers` right after `swig`.

Once you have finished making changes to the wrapper, check to make sure 
everything is operational by running the Python unit tests in the `tests/` 
directory. From the top-level directory of the CCL sources, and assuming that 
you have recompiled and reinstalled the C library and the Python wrapper, run:
  `$ python tests/run\_tests.py`
This may take some time to run in its entirety. If you changed the API, you may 
have to modify the tests to account for this. You should also add your own 
tests for any new functions or behaviors that were added.

Occasionally, modifications made correctly as described above will still not 
function properly. This is because python installation files are not being 
properly overwritten, and the interpreter is getting confused which version to 
use. At this point it is often best to resort to the "nuclear option" of 
deleting all python files related to pyccl and starting from scratch. The 
procedure is as follows:

 1. Run clean scripts
    
   - Execute `make clean`

   - Execute `python setup.py uninstall`

 2. Delete built/installed python files

   - Execute `rm -r build/` to delete the python build directory

   - Execute `rm pyccl/ccl_wrap.c pyccl/ccllib.py` to delete the autogenerated swig files

   - Check your `~/.local/lib` directory for any pyccl files (they will be in one of the python directories), 
     and delete them.

   - Similarly for files in your `/usr/lib/` and `/usr/local/lib` directories

   - Still not working? You can execute `python setup.py install [args] --record files.txt`. 
     The file `files.txt` will contain a list of all files written when calling the install script with
     arguments `args`. Perhaps there are files somewhere else on your system! 

 3. You can now reinstall the package in the normal way, starting with the 
    C code and then the python code. 

API Changes and Stability
--------------------------------------------
An API breakage is anything that modifies the name or call signature of an 
existing public-facing function or variable (e.g. any function with the prefix 
`ccl_` in the C headers). Changes to keyword arguments of Python functions also 
fit this description, unless they are explicitly made backwards-compatible. 
Significant changes to the behaviour of a function also count as an API 
breakage (e.g. if an assumption is changed that causes different results to be 
produced by default). Fixing a bug that affects the output of a function does 
not count as an API breakage, unless the fix modifies function names and 
arguments too.

If the API is broken, the CCL version number must be incremented (e.g. from 
v0.1 to v0.2), and a new tagged release should be made shortly after the 
changes are pushed to master. The new release must include notes in 
`CHANGELOG.md` that describe how the API has been changed. The aim is to make 
it clear to users how their CCL-based code might be affected if they update to 
the latest version, especially if the update is likely to break something. All 
API changes should be discussed with the CCL team before being pushed to the 
`master` branch.

There is not yet any absolute need to preserve backwards-compatibility between 
point releases (e.g. v0.2 does not need to support everything that v0.1 did). 
This will change as the code matures, though. After v1.0 is released, we expect 
to maintain backwards compatibility for all releases with the same major 
version number, i.e. v1.1 should support all the same behaviors that v1.0 
supported, plus whatever new additions are made.

Compiling the CCL note
--------------------------------------------
After making changes to the library, you should document them in the CCL note. 
The note is found in the directory `doc/0000-ccl_note/`. To compile the note, 
type `make` in that directory. If you do not have `pip` installed, please edit 
the `Makefile` to agree with your setup. If you do not have admin permission,
then you will need to set up a virtual environment to install the note. This 
is done as follows. Set up the virtual environment once:
  ```sh
  $ virtualenv CCL
  ```
and after a new login:
  ```sh
  $ source CCL/activate
  $ make
  ```
If you need to modify the note, the files to modify are:
  
 - `authors.csv`: To document your contribution.

 - `main.tex`: To detail the changes to the library.

 - `main.bib`: To add new references.

Travis-CI
--------------------------------------------

Travis-CI is a continuous integration tool that reads the file `.travis.yml` 
and performs the steps described there in a virtual environment. More details 
can be found here: https://docs.travis-ci.com/user/getting-started/

Every time you perform a commit Travis-CI will automatically try to build the 
libraries with your new changes and run the unit tests. You can check the 
status of your build here: https://travis-ci.org/LSSTDESC/CCL/builds. If you 
click in your build you will find more information about its status and a log 
describing the process. If your build errors or fails, you can scroll through 
the log to find out what went wrong. If your additions require new dependencies 
make sure that you include them in `.travis.yml`.
