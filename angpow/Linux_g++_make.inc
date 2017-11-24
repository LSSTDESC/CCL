#GCCDIR = /opt/rh/devtoolset-3/root/usr/bin/


#  Flag preprocesseur
CPPFLAGS = -DLinux 
#  Def compilateur C et flags
#CC = $(GCCDIR)gcc
CC = gcc
# Replace [CNFPHFLF] by specific compilation flags below (done by configure)
CFLAGS = -Wall -Wpointer-arith -Wmissing-prototypes -O  -fPIC 
#  Def compilateur C++ et flags
#CXX =  $(GCCDIR)g++
CXX =  c++
#  Flag de warning -Wsynth NON inclus par defaut (04/2007)
CXXFLAGS = -Wall -Wpointer-arith -O3 -fPIC -std=c++11 -fopenmp  -fno-common  -Wall -march=native -ffast-math -m64
# flags specifiques pour templates repository...
CXXTEMPFLG =
#  Compilo fortran
#FC = g77
#FC = $(GCCDIR)gfortran
FC = gfortran
FFLAGS =  
# Pour fabriquer les lib .a
AR = ar
ARFLAGS = -rcs

#  Ordres de compilation
CCOMPILE = $(CC) $(CPPFLAGS) $(CFLAGS) -c 
CXXCOMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c 
FCCOMPILE = $(FC) $(FFLAGS) -c 

#  Extension de nom pour les librairies partagees
SLEXT = so
#  Fabrication de librairies partagees 
CMDSHLCXX = $(CXX) $(CXXFLAGS) $(CXXTEMPFLG) -shared 


LIBFORT = -lgfortran


#  Commande de link pour creation dexecutables
CXXLINK = $(CXX) $(CXXFLAGS)
CLINK = $(CC) $(CFLAGS)
FCLINK = $(FC) $(FFLAGS)

#  Commande de link pour creation de module pouvant etre charge dynamiquement
CXXDLL = $(CXX) $(CXXFLAGS) -shared
