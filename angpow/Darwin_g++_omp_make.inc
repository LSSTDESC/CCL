

#  Flag preprocesseur
CPPFLAGS = -DDarwin 
#  Def compilateur C et flags
CC = gcc
# Replace [CNFPHFLF] by specific compilation flags below (done by configure)
CFLAGS = -fno-common  -O -fPIC
#  With optimization flags 
#  CFLAGS = -fno-common -g -fastf -mtune=G5 -fPIC
#  Def compilateur C++ et flags
CXX =  g++
#JEC add -fopenmp -O3 -Wa,-q
CXXFLAGS = -fno-common  -O3 -fPIC -fopenmp  -Wa,-q -Wall -Wpedantic -std=c++11 -march=native -ffast-math -m64
#CXXFLAGS = -fno-common -g -O3 -fPIC -Wa,-q -Wall -std=c++11
# flags specifiques pour templates repository...
CXXTEMPFLG =
#  Compilo fortran
FC = gfortran
FFLAGS =  
# Pour fabriquer les lib .a
AR = libtool 
ARFLAGS = -static -o 


#  Ordres de compilation
CCOMPILE = $(CC) $(CPPFLAGS) $(CFLAGS) -c 
CXXCOMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c 
FCCOMPILE = $(FC) $(FFLAGS) -c 

#  Extension de nom pour les librairies partagees
SLEXT = dylib
#  Fabrication de librairies partagees 
CMDSHLCXX = $(CXX) $(CXXFLAGS) $(CXXTEMPFLG) -dynamiclib 


LIBFORT = 

#  Commande de link pour creation d'executables
CXXLINK = $(CXX) $(CXXFLAGS) -bind_at_load
#  Commande de link pour creation de module pouvant etre charge dynamiquement
CXXDLL = $(CXX) $(CXXFLAGS) -bundle 
