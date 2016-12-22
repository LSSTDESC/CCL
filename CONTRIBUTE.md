Summary of changes wrt the pre-autotools version:
---------------------------------------------------------
 - No Makefile: everything is now defined in
     Makefile.am
     include/Makefile.am
     class/Makefile.am
 - When adding new files to src, the new file should
   be listed under libccl_la_SOURCES in Makefile.am
 - When adding new files to include, the new file 
   should be listed under include_HEADERS in
   include/Makefile.am
 - When adding new unit test files, they should be
   listed under check_ccl_SOURCES in Makefile.am
 - Any other new files that should be included with
   the library (e.g. documentation files) should be
   listed under EXTRA_DIST in Makefile.am.
 - The file configure.ac may have to be modified when
   adding new features, but not too often.
 - After making changes to any of these files, developers
   should run:
      $> autoreconf -i
   and commit any modified files (some files will be
   automatically modified by this command).
 - From now on, to install the library, users will run:
     $> configure <options>
     $> make
     $> make install
 - Once the library is installed, all unit tests are
   run by typing:
     $> make check
 - A tarball with all files needed to install the library
   can be created automatically by typing:
     $> make dist
