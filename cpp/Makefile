# Standalone C++ triangle field. No fast-math. Optional CPUSET affinity.
CXX ?= g++
CPUSET ?=
CXXFLAGS ?= -pthread -std=c++17 -O2 -fno-fast-math -Wall -Wextra
CPPFLAGS ?= -Iinclude
LDFLAGS ?=
LDLIBS ?= -ldl
BUILD ?= build
RUN ?= $(if $(CPUSET),taskset -c $(CPUSET),)
TIMEOUT ?= 600
MEMKB ?= 8388608

ROOT := $(abspath $(CURDIR)/..)
NATIVE_SRC := $(ROOT)/fortran/native
EXACT_SRC := $(ROOT)/fortran/triangle_exact_geometry.c

.PHONY: all test test-native clean example libtan

all: $(BUILD)/triangle_cli $(BUILD)/native_cli $(BUILD)/test_triangle $(BUILD)/example

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/triangle.o: src/triangle.cpp include/triangle_pm/triangle.hpp | $(BUILD)
	$(RUN) $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c src/triangle.cpp -o $@

$(BUILD)/native.o: src/native.cpp include/triangle_pm/native.hpp include/triangle_pm/triangle.hpp | $(BUILD)
	$(RUN) $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c src/native.cpp -o $@

$(BUILD)/triangle_cli: $(BUILD)/triangle.o src/triangle_cli.cpp | $(BUILD)
	$(RUN) $(CXX) $(CXXFLAGS) $(CPPFLAGS) src/triangle_cli.cpp $(BUILD)/triangle.o -o $@ $(LDFLAGS)

$(BUILD)/native_cli: $(BUILD)/triangle.o $(BUILD)/native.o src/native_cli.cpp | $(BUILD)
	$(RUN) $(CXX) $(CXXFLAGS) $(CPPFLAGS) src/native_cli.cpp $(BUILD)/triangle.o $(BUILD)/native.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/test_triangle: $(BUILD)/triangle.o $(BUILD)/native.o tests/test_triangle.cpp | $(BUILD)
	$(RUN) $(CXX) $(CXXFLAGS) $(CPPFLAGS) tests/test_triangle.cpp $(BUILD)/triangle.o $(BUILD)/native.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/example: $(BUILD)/triangle.o examples/example.cpp | $(BUILD)
	$(RUN) $(CXX) $(CXXFLAGS) $(CPPFLAGS) examples/example.cpp $(BUILD)/triangle.o -o $@ $(LDFLAGS)

test: $(BUILD)/test_triangle $(BUILD)/triangle_cli $(BUILD)/example
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	$(RUN) bash -c 'ulimit -v $(MEMKB); timeout $(TIMEOUT) $(BUILD)/test_triangle'
	OMP_NUM_THREADS=1 $(RUN) $(BUILD)/example >/dev/null
	$(RUN) $(BUILD)/triangle_cli 0 1 0.8 0 0 0 1.2 0 0 0.2 0.9 0 0.4 0.3 0.5 >/dev/null

$(BUILD)/libtan.so: | $(BUILD)
	$(RUN) gcc -O3 -fno-fast-math -std=c11 -fPIC -c $(EXACT_SRC) -o $(BUILD)/triangle_exact_geometry.o
	$(RUN) gfortran -O3 -fno-fast-math -std=f2018 -fPIC -ffree-line-length-none -J$(BUILD) -c $(NATIVE_SRC)/tan_jet.f90 -o $(BUILD)/tan_jet.o
	$(RUN) gfortran -O3 -fno-fast-math -std=f2018 -fPIC -ffree-line-length-none -J$(BUILD) -I$(BUILD) -c $(NATIVE_SRC)/tan_mod.f90 -o $(BUILD)/tan_mod.o
	$(RUN) gfortran -O3 -fno-fast-math -shared -o $(BUILD)/libtan.so $(BUILD)/tan_jet.o $(BUILD)/tan_mod.o $(BUILD)/triangle_exact_geometry.o -lgmp

$(BUILD)/triangle.cache: | $(BUILD)
	printf '%s\n' 'TAN_CACHE_V1' '1 3' \
	  '0 0 0 0 0 1.08' \
	  '1 0 0 0 1.2 0 0 0.8' \
	  '1 1.2 0 0 0.2 0.9 0 0.8' \
	  '1 0.2 0.9 0 0 0 0 0.8' > $@

test-native: $(BUILD)/test_triangle $(BUILD)/native_cli $(BUILD)/libtan.so $(BUILD)/triangle.cache
	OMP_NUM_THREADS=1 $(RUN) $(BUILD)/native_cli $(abspath $(BUILD)/libtan.so) $(abspath $(BUILD)/triangle.cache) 0.4 0.3 0.5 1 0 1 >/dev/null
	TRIANGLE_NATIVE_LIB=$(abspath $(BUILD)/libtan.so) TRIANGLE_NATIVE_CACHE=$(abspath $(BUILD)/triangle.cache) \
	OMP_NUM_THREADS=1 $(RUN) bash -c 'ulimit -v $(MEMKB); timeout $(TIMEOUT) $(BUILD)/test_triangle'

example: $(BUILD)/example
	$(RUN) $(BUILD)/example

clean:
	rm -rf $(BUILD)
	rm -f tan_jet_mod.mod tan_mod.mod
