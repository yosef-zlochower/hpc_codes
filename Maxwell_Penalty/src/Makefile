CC       = mpicc
CXX      = mpicxx
CFLAGS   = -Wall -O3 -ffast-math -g
CXXFLAGS = -Wall -O3 -ffast-math -g -std=c++17

SRCS=comm.c domain.c gf.c rk4.c maxwell_eqs.c io.c HDF5BinaryWrite.c timer.c numerical.c analytic_solutions.c driver.c
OBJS=comm.o domain.o gf.o rk4.o maxwell_eqs.o io.o HDF5BinaryWrite.o timer.o numerical.o analytic_solutions.o driver.o
HDRS= comm.h derivatives.h domain.h gf.h io.h rk4.h parameter.h maxwell_eqs.h simple_maxwell.h

# Core objects shared by tests (everything except maxwell_eqs.o and driver.o)
CORE_OBJS=comm.o domain.o gf.o rk4.o io.o HDF5BinaryWrite.o timer.o numerical.o analytic_solutions.o

TEST_PROGS = tests/test_sync tests/test_rk4 tests/test_topology

# ── Default target ────────────────────────────────────────────────────────────

maxwell_system: $(OBJS) parameter.o maxwell_parameters.o
	$(CXX) -g -Wl,-rpath=$(HDF5_LIB_DIR) -lhdf5 -lhdf5_hl -fopenmp $(OBJS) parameter.o maxwell_parameters.o -o $@ -lm

# ── C source rules ────────────────────────────────────────────────────────────

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $<

tests/%.o: tests/%.c $(HDRS)
	$(CC) $(CFLAGS) -I. -o $@ -c $<

# ── C++ source rules (parameter parsing) ─────────────────────────────────────

parameter.o: parameter.cc parameter.h toml.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

maxwell_parameters.o: maxwell_parameters.cc parameter.h toml.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

# ── Test executables ─────────────────────────────────────────────────────────

tests/test_sync: tests/test_sync.o $(CORE_OBJS)
	$(CC) -g $^ -Wl,-rpath=$(HDF5_LIB_DIR) -lhdf5 -lhdf5_hl -o $@ -lm

tests/test_rk4: tests/test_rk4.o $(CORE_OBJS)
	$(CC) -g $^ -Wl,-rpath=$(HDF5_LIB_DIR) -lhdf5 -lhdf5_hl -o $@ -lm

tests/test_topology: tests/test_topology.o domain.o timer.o
	$(CC) -g $^ -o $@ -lm

# ── Test runner ──────────────────────────────────────────────────────────────

tests: $(TEST_PROGS)
	cd tests && bash run_tests.sh

# ── Generated-header verification ────────────────────────────────────────────
#
# simple_maxwell.h is produced by generate_ccode.py (SymPy).  It is checked
# in so the build does not require a SymPy install, but that means the two
# can drift.  `make check-generated` re-runs the generator in a temp
# directory and diffs its output against the committed header; a non-zero
# exit (and a unified diff printed to stderr) flags drift.
#
# To regenerate for real after editing generate_ccode.py:
#   python3 generate_ccode.py
PYTHON ?= python3
check-generated:
	@tmp=$$(mktemp -d); \
	 cp generate_ccode.py "$$tmp/"; \
	 (cd "$$tmp" && $(PYTHON) generate_ccode.py); \
	 if diff -u simple_maxwell.h "$$tmp/simple_maxwell.h" > /dev/null; then \
	     echo "check-generated: simple_maxwell.h matches generate_ccode.py output."; \
	     rm -rf "$$tmp"; \
	 else \
	     echo "check-generated: FAIL -- simple_maxwell.h is out of date." >&2; \
	     echo "Re-run 'python3 generate_ccode.py' and commit the result." >&2; \
	     echo "--- committed -------------------------------------------" >&2; \
	     echo "+++ generator output ------------------------------------" >&2; \
	     diff -u simple_maxwell.h "$$tmp/simple_maxwell.h" >&2 || true; \
	     rm -rf "$$tmp"; \
	     exit 1; \
	 fi

clean:
	-rm $(OBJS) parameter.o maxwell_parameters.o maxwell_system \
	    tests/test_sync.o tests/test_rk4.o tests/test_topology.o \
	    $(TEST_PROGS)

.PHONY: tests check-generated clean
