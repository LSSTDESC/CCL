CCLINCLUDE = ../include


all: ccl_sample_run

clean:
	rm -f ccl_sample_run
	rm -rf ccl_sample_run.dSYM

ccl_sample_run:
	gcc -Wall -Wpedantic -g -I$(CCLINCLUDE) -std=gnu99 -fPIC ccl_sample_run.c -o ccl_sample_run  -lgsl -lgslcblas -lm -lccl
