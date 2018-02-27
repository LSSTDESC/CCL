from __future__ import print_function
import os
import sys
import getopt
import fileinput

#Help message
help_msg= "Usage : angpow_install.py [options]\n"
help_msg+="Options :\n"
help_msg+=" -h, --help : This help\n"
#help_msg+=" -c, --c-comp= : which C compiler to use (default: gcc)\n"
help_msg+=" -c, --clean= : Clean angpow installation (default: False)\n"
#help_msg+=" -omp, --enable-openmp: enable OpenMP (default: yes if Linux)\n"
#Current directory
dir_path=os.path.dirname(os.path.realpath(__file__))

#Header for log file
flog=open("log_angpow_install","w")
flog.write("ANGPOW install logfile\n\n")
flog.close()

#Final cleanup
def cleanup(full_cleanup=False) :
    os.chdir(dir_path)
    os.system('rm -f v0.2.tar.gz')
    os.system('rm -f log_angpow_install')
    if full_cleanup :
        os.system('rm -r angpow')

#Executes UNIX command, checks for success and exits if failure
def check_command(command) :
    st=os.system(command+" >>log_angpow_install 2>&1")
    if st :
        print(" Command "+command[:10]+"... failed.")
        print(" See log file contents")
        os.system("cat log_angpow_install")
        cleanup(full_cleanup=True)
        sys.exit(1)

#Makes necessary modifications to ANGPOW's makefile
#def mod_makefile(cname,ompflag) :
#    for line in fileinput.FileInput("Makefile",inplace=1) :
#        if line.startswith('CC ') :
#            print("CC       = %s\n"%cname,end="")
#        #elif line.startswith('OMPFLAG ') :
#        #    print("OMPFLAG  =%s\n"%ompflag,end="")
#        else:
#            print("%s"%line,end="")
        
#Check input arguments
try :
    opts, args = getopt.getopt(sys.argv[1:],"hc",["help","clean"])
except getopt.GetoptError:
    print(help_msg)
    sys.exit(1)
    
clean = False
ompflag='-fopenmp'
for opt,arg in opts :
    if opt in ("-h","--help") :
        print(help_msg)
        sys.exit(1)
    elif opt in ("-c","--clean") :
        clean=True
    #elif opt in ("-omp","--enable-openmp"):
    #    ompflag='-fopenmp'

if(clean):
    print("Cleanup")
    cleanup(clean)
else:
    #Actual installation
    print("Downloading Angpow...")
    if sys.platform.startswith('linux') :
        check_command('wget -q https://github.com/LSSTDESC/Angpow4CCL/archive/v0.2.tar.gz')
        print("Unpacking...")
        check_command('tar -xvf v0.2.tar.gz ')
    if sys.platform.startswith('darwin') :
        check_command('curl -L https://github.com/LSSTDESC/Angpow4CCL/archive/v0.2.tar.gz | tar xz')
    check_command('mv Angpow4CCL-0.2 angpow')

    print("Compiling...")
    os.chdir('angpow')
    if not os.path.exists('Objs'): os.mkdir('Objs')
    if not os.path.exists('lib'): os.mkdir('lib')
    #mod_makefile(c_comp,ompflag)
    check_command('make lib')

    print("Cleanup")
    cleanup()
