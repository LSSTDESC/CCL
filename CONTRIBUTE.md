Reviewing a pull request (PR) on github
---------------------------------------
1. Checkout the branch.
2. Make sure you can install the C library.
3. Make sure you can install the python module.
4. Make sure the C unit tests pass (i.e. run "make check" successfully).
5. Make sure the python tests pass (i.e. run "python run_tests.py" from the "tests" directory).
6. Look at the code (see "Files changed" on the top of this page) and check that the changes make sense to you.
7. If new science has been implemented, and if possible, try to compare the output of the code against your own predictions. Possibly, ask the developer to implement a unit test for it.
8. Make sure that the libraries install successfully and the unit tests pass at Travis-CI.

Things to do if you are adding new features to the CCL C lib
------------------------------------------------------------
 1. If you haven't created a new files (i.e. you only edited
    an existing one), you don't need to do any of this.
 2. When adding a new source file (.c), put it in src. The
    the new file should be listed under libccl_la_SOURCES in
    Makefile.am.
 3. When adding a new header file (.h), put it in include.
    The new file should be listed under include_HEADERS in
    include/Makefile.am. It should also have doxygen
    compatible documentation.
 4. When adding new unit test files, they should be listed
    under check_ccl_SOURCES in Makefile.am
 5. Any other new files that should be included with the
    library (e.g. documentation files) should be listed
    under EXTRA_DIST in Makefile.am.
 6. The file configure.ac may have to be modified when 
    adding new features (e.g. adding a new dependency), but
    not too often.
 7. After making changes to any of these files, developers
    should run:
      $> autoreconf -i
    and commit any modified files (some files will be
    automatically modified by this command).

More autotools fun:
 - To install the library, users will run:
     $> configure <options>
     $> make
     $> make install
 - Once the library is compiled, all unit tests are run by
   typing:
     $> make check
 - A tarball with all files needed to install the library
   can be created automatically by typing:
     $> make dist

To view the doxygen documentation, open any .html file in the html/ 
directory. To refresh the docs to reflect new changes, run 
`doxygen` in the main directory (assuming you already have it installed).

Modifying the Python wrapper
---------------------------------------------------------
If you make changes to the public CCL API, you will also have to update the 
Python wrapper. The most important task is to make sure your new or modified 
functions are properly exposed to the user through the wrapper. This should be 
quite straightforward in most instances.

To do this, examine the following files (where <modulename> is the name of the 
CCL module that your new/modified function belongs to; e.g. 'background', if the 
function is in ccl\_background.c):

 - pyccl/<modulename>.i: These are the SWIG interface files, and mostly handle 
   things like vectorizing C functions. This is usually done by adding a loop 
   over the indices of the input array, and by providing a function call 
   signature that can be used to accept numpy arrays. See function names ending 
   in '\_vec' in the interface file for examples. If you changed the call 
   signature of a function, there is a good chance you'll have to also change 
   it in the corresponding interface file.
 - pyccl/<modulename>.py: These are Python wrappers around the auto-generated 
   SWIG wrapper (which is only exposed through the pyccl.lib namespace). They 
   are there to present a more 'Pythonic' interface to the user, for example by 
   wrapping CCL structs in more user-friendly classes, automatically handling 
   memory management logic, and doing type checking and error checking.
   These files are where you should define classes to manage CCL objects (see 
   the Parameters and Cosmology classes in pyccl/core.py for example), and 
   where you should provide easy-to-use wrappers around more complicated 
   functions provided through the basic SWIG wrapper. Ideally, you would also 
   provide some type checking and error checking code in this part of the 
   wrapper.
 - pyccl/\_\_init\_\_.py: This file is used to define the top-level user interface 
   of the pyccl module. You should make sure that your new/modified functions 
   are imported.

If you added new functions to the interface, you should also take a look in 
the following files:

 - pyccl/ccl.i: This is the root interface file, and includes all of the 
   interface files. If you added a new module to CCL, you will probably need to 
   create a <modulename>.i file for it. Once you've done that, include it in 
   ccl.i and it should be automatically picked up by the build system.
 - pyccl/pyutils.py: This file contains convenience functions for calling CCL 
   functions that take numpy array input/output arguments, and for passing the 
   Cosmology() class as an argument to a function in the SWIG wrapper. Use 
   these if possible, instead of writing your own wrapper code for individual 
   functions.

After these changes have been made, you should recreate pyccl/ccl_wrap.c and
pyccl/ccllib.py by running swig:

~$ swig -python -threads -o pyccl/ccl_wrap.c pyccl/ccl.i


Once you have finished making changes to the wrapper, check to make sure 
everything is operational by running the Python unit tests in the tests/ 
directory. From the top-level directory of the CCL sources, and assuming that 
you have recompiled and reinstalled the C library and the Python wrapper, run:

  $ python tests/run\_tests.py

This may take some time to run in its entirety. If you changed the API, you may 
have to modify the tests to account for this. You should also add your own 
tests for any new functions or behaviors that were added.

Occasionally, modifications made correctly as described above will still not function properly.
This is because python installation files are not being properly overwritten, and 
the interpreter is getting confused which version to use. At this point it is often best to resort 
to the "nuclear option" of deleting all python files related to pyccl and starting from 
scratch. The procedure is as follows:
1. Run clean scripts
    
1.1 Execute `make clean`

1.2 Execute `python setup.py uninstall`

2. Delete built/installed python files

2.1 Execute `rm -r build/` to delete the python build directory

2.2 Execute `rm pyccl/ccl_wrap.c pyccl/ccllib.py` to delete the autogenerated swig files

2.3 Check your `~/.local/lib` directory for any pyccl files (they will be in one of the python directories), and delete them.

2.4 Similarly for files in your `/usr/lib/` and `/usr/local/lib` directories

2.5 Still not working? You can exectute `python setup.py install [args] --record files.txt`. 
The file `files.txt` will contain a list of all files written when calling the install script with
arguments `args`. Perhaps there are files somewhere else on your system! 

3. You can now reinstall the package in the normal way, starting with the 
C code and then the python code. 

Compiling the CCL note
--------------------------------------------
After making changes to the library, you should document them in the
CCL note. The note is found in the directory doc/0000-ccl_note/.
To compile the note, type 

  $make

in that directory. If you do not have pip installed, please edit the
Makefile to agree with your setup. If you do not have admin permission,
then you will need to setup a virtual environment to install the note.
This is done as follows. Set up the virtual environment once:
  
  $virtualenv CCL

and after a new login

  $source CCL/activate
  $make

If you need to modify the note, the files to modify are:
  
  -authors.csv: to document your contribution

  -main.tex: to detail the changes to the library

  -main.bib: to add new references

Travis-CI
--------------------------------------------

Travis-CI is a continuous integration tool that reads the file `.travis.yml` and performs
the steps described there in a virtual environment. More details can be found here: https://docs.travis-ci.com/user/getting-started/

Every time you perform a commit Travis-CI will automatically try to build the libraries with your new changes and run the unit tests. You can check the status of your build here: https://travis-ci.org/LSSTDESC/CCL/builds. If you click in your build you will find more information about its status and a log describing the process. If your build errored or failed you can scroll through the log to find out what went wrong. If your additions require new dependencies make sure that you include them in `.travis.yml`.
