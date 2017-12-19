from __future__ import print_function
import os
import sys
import getopt
import fileinput

#Help message
help_msg= "Usage : class_install.py [options]\n"
help_msg+="Options :\n"
help_msg+=" -h, --help : This help\n"
help_msg+=" -c, --c-comp= : which C compiler to use (default: gcc)\n"
help_msg+=" -omp, --enable-openmp: enable OpenMP (default: yes if Linux)\n"
#Current directory
dir_path=os.path.dirname(os.path.realpath(__file__))

#Header for log file
flog=open("log_class_install","w")
flog.write("CLASS install logfile\n\n")
flog.close()

#Final cleanup
def cleanup(full_cleanup=False) :
    os.chdir(dir_path)
    os.system('rm -f v2.6.3.tar.gz')
    os.system('rm -f log_class_install')
    if full_cleanup :
        os.system('rm -r class')

#Executes UNIX command, checks for success and exits if failure
def check_command(command) :
    st=os.system(command+" >>log_class_install 2>&1")
    if st :
        print(" Command "+command[:10]+"... failed.")
        print(" See log file contents")
        os.system("cat log_class_install")
        cleanup(full_cleanup=True)
        sys.exit(1)

#Makes necessary modifications to CLASS's makefile
def mod_makefile(cname,ompflag) :
    for line in fileinput.FileInput("Makefile",inplace=1) :
        if line.startswith('CC ') :
            print("CC       = %s\n"%cname,end="")
        elif line.startswith('OMPFLAG ') :
            print("OMPFLAG  =%s\n"%ompflag,end="")
        else:
            print("%s"%line,end="")
        
#Check input arguments
try :
    opts, args = getopt.getopt(sys.argv[1:],"hc:",["help","c_comp="])
except getopt.GetoptError:
    print(help_msg)
    sys.exit(1)
    
c_comp='gcc'
ompflag=''
for opt,arg in opts :
    if opt in ("-h","--help") :
        print(help_msg)
        sys.exit(1)
    elif opt in ("-c","--c_comp") :
        c_comp=arg
    elif opt in ("-omp","--enable-openmp"):
        ompflag='-fopenmp'

#Actual installation        
print("Downloading class...")
if sys.platform.startswith('linux') :
    check_command('wget -q https://github.com/lesgourg/class_public/archive/v2.6.3.tar.gz')
    print("Unpacking...")
    check_command('tar -xvf v2.6.3.tar.gz ')
if sys.platform.startswith('darwin') :
    check_command('curl -L https://github.com/lesgourg/class_public/archive/v2.6.3.tar.gz | tar xz')
check_command('mv class_public-2.6.3 class')

print("Compiling...")
os.chdir('class')
mod_makefile(c_comp,ompflag)
check_command('make libclass.a')
#if sys.platform.startswith('linux') :
#    comp_string=c_comp+' -shared -o libclass.so -Wl,--whole-archive libclass.a -Wl,--no-whole-archive -lgomp >> ./log_class_install ; '
#elif sys.platform.startswith('darwin') :
#    comp_string=c_comp+" -fpic -shared -o libclass.dylib -Wl,-all_load libclass.a -Wl,-noall_load"
#else :
#    raise OSError("Can't figure out your system : "+sys.platform)
#check_command(comp_string)

print("Cleanup")
cleanup()
