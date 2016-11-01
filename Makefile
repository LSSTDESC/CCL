CC=gcc
CFLAGS=-Wall -Wpedantic -g -O0 -Iinclude -std=c99 -fPIC
CFLAGS+=-I/opt/local/include
LDFLAGS=-lgsl -lgslcblas   -lm -Lclass -lclass


OBJECTS=src/ccl_core.o src/ccl_utils.o src/ccl_error.o src/ccl_power.o src/ccl_cls.o src/ccl_placeholder.o src/ccl_background.o

TESTS=tests/ccl_test.c tests/ccl_test_utils.c tests/ccl_test_params.c tests/ccl_test_distances.c
#
# Tests to include at some point:
# tests/ccl_test_power.c  tests/ccl_test_bbks.c
LIB=lib/libccl.a
DYLIB=lib/libccl.so
INC_CCL=

all: $(LIB) $(DYLIB)

$(LIB): $(OBJECTS) class
	ar rc $(LIB) $(OBJECTS)

$(DYLIB): $(OBJECTS)
	$(CC) -shared -o $(DYLIB) $(OBJECTS) $(CFLAGS) $(LDFLAGS)

class:
	cd class; $(MAKE)

test: $(TESTS) $(LIB)
	$(CC) $(CFLAGS) $(TESTS) -o tests/ccl_test -Llib -lccl $(LDFLAGS)
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:lib/ tests/ccl_test

src/%.o: src/%.c
	$(CC) -c $(CFLAGS) $< -o $@


clean:
	rm -rf *.dSYM *.o *.a tests/ccl_test test_core_cosmo src/*.o lib/*.a lib/*.so lib/*.dSYM  tests/*.dSYM
	cd class; $(MAKE) clean

.PHONY: all tests clean class
