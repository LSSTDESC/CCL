CC=gcc
CFLAGS=-Wall -Wpedantic -g -O0 -Iinclude -std=c99 -fPIC
CFLAGS+=-I/home/damonge/include
LDFLAGS=-lgsl -lgslcblas   -lm


OBJECTS=src/ccl_core.o src/ccl_utils.o src/ccl_power.o src/ccl_placeholder.o src/ccl_background.o
TESTS=tests/ccl_test_utils tests/ccl_test_power tests/ccl_test_distances
LIB=lib/libccl.a
DYLIB=lib/libccl.so
INC_CCL=

all: $(LIB) $(DYLIB) test

$(LIB): $(OBJECTS)
	ar rc $(LIB) $(OBJECTS)

$(DYLIB): $(OBJECTS)
	$(CC) -shared -o $(DYLIB) $(OBJECTS) $(CFLAGS) $(LDFLAGS)


test: $(TESTS)
	@echo
	@echo "Running test programs"
	@echo "---------------------"
	tests/ccl_test_utils > /dev/null
	tests/ccl_test_power > /dev/null
	tests/ccl_test_distances > /dev/null
	@echo "---------------------"
	@echo

tests/% : tests/%.c $(LIB)
	$(CC)  $(CFLAGS) $< -o $@ -Llib -lccl $(LDFLAGS) 

src/%.o: src/%.c
	$(CC) -c $(CFLAGS) $< -o $@


clean:
	rm -rf *.dSYM *.o *.a $(TESTS) test_core_cosmo src/*.o lib/*.a lib/*.so lib/*.dSYM  tests/*.dSYM

.PHONY: all tests clean test