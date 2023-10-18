GSLINC = /opt/local/include
GSLLIB = /opt/local/lib

emu.exe: emu.c
	gcc -L$(GSLLIB) -I$(GSLINC) emu.c -lgsl -lgslcblas -lm -o emu.exe

clean:
	/bin/rm -f *.o