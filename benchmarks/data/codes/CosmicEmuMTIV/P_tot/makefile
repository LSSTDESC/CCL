GSLINC = /usr/local/include
GSLLIB = /usr/local/lib

emu.exe: emu.c
	gcc -L$(GSLLIB) -I$(GSLINC) emu.c -lgsl -lgslcblas -lm -o emu.exe

clean:
	/bin/rm -f *.o