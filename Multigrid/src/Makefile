CC       = mpicc
CXX      = mpicxx
CFLAGS   = -Wall -O3 -ffast-math -g
CXXFLAGS = -Wall -O3 -ffast-math -g -std=c++17

SRCS = domain.c  gf.c timer.c comm.c multigrid.c gauss_seidel.c io.c
OBJS = domain.o  gf.o  timer.o comm.o multigrid.o gauss_seidel.o io.o
HDRS = domain.h   gf.h multigrid.h  timer.h comm.h io.h

TEST_PROGS = tests/test_domain_2d tests/test_domain_3d \
             tests/test_child_2d   tests/test_child_3d  \
             tests/test_project_2d tests/test_project_3d \
             tests/test_prolong_2d tests/test_prolong_3d \
             tests/test_restrict_nl_2d tests/test_restrict_nl_3d \
             tests/test_prolong_nl_2d  tests/test_prolong_nl_3d

TEST_OBJS = tests/test_domain_2d.o tests/test_domain_3d.o \
            tests/test_child_2d.o   tests/test_child_3d.o  \
            tests/test_project_2d.o tests/test_project_3d.o \
            tests/test_prolong_2d.o tests/test_prolong_3d.o \
            tests/test_restrict_nl_2d.o tests/test_restrict_nl_3d.o \
            tests/test_prolong_nl_2d.o  tests/test_prolong_nl_3d.o

all: driver_multigrid

# ── C source rules ────────────────────────────────────────────────────────────

%.o: %.c  $(HDRS)
	$(CC)  -c $(CFLAGS)  $<

tests/%.o: tests/%.c  $(HDRS)
	$(CC)  -c $(CFLAGS)  -I.  -o $@  $<

gauss_seidel.o: gauss_seidel.c gauss_seidel.h $(HDRS)
	$(CC) -c $(CFLAGS) $<

multigrid.o: multigrid.c gauss_seidel.h $(HDRS)
	$(CC) -c $(CFLAGS) $<

driver_multigrid.o: driver_multigrid.c gauss_seidel.h multigrid.h parameter.h $(HDRS)
	$(CC) -c $(CFLAGS) $<

# ── C++ source rules ──────────────────────────────────────────────────────────

parameter.o: parameter.cc parameter.h toml.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

multigrid_parameters.o: multigrid_parameters.cc parameter.h toml.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

# ── Executables ───────────────────────────────────────────────────────────────

# Link with mpicxx so the C++ runtime is included alongside the MPI C objects
driver_multigrid: driver_multigrid.o parameter.o multigrid_parameters.o $(OBJS)
	$(CXX) -g $^ -o $@ -lm


# ── Tests ─────────────────────────────────────────────────────────────────────

tests/test_domain_2d: $(OBJS) tests/test_domain_2d.o
	$(CC) -g $(OBJS)  tests/test_domain_2d.o -o $@ -lm

tests/test_domain_3d: $(OBJS) tests/test_domain_3d.o
	$(CC) -g $(OBJS)  tests/test_domain_3d.o -o $@ -lm

tests/test_child_2d: $(OBJS) tests/test_child_2d.o
	$(CC) -g $(OBJS) tests/test_child_2d.o -o $@ -lm

tests/test_child_3d: $(OBJS) tests/test_child_3d.o
	$(CC) -g $(OBJS) tests/test_child_3d.o -o $@ -lm

tests/test_project_2d: $(OBJS) tests/test_project_2d.o
	$(CC) -g $(OBJS) tests/test_project_2d.o -o $@ -lm

tests/test_project_3d: $(OBJS) tests/test_project_3d.o
	$(CC) -g $(OBJS) tests/test_project_3d.o -o $@ -lm

tests/test_prolong_2d: $(OBJS) tests/test_prolong_2d.o
	$(CC) -g $(OBJS) tests/test_prolong_2d.o -o $@ -lm

tests/test_prolong_3d: $(OBJS) tests/test_prolong_3d.o
	$(CC) -g $(OBJS) tests/test_prolong_3d.o -o $@ -lm

tests/test_restrict_nl_2d: $(OBJS) tests/test_restrict_nl_2d.o
	$(CC) -g $(OBJS) tests/test_restrict_nl_2d.o -o $@ -lm

tests/test_restrict_nl_3d: $(OBJS) tests/test_restrict_nl_3d.o
	$(CC) -g $(OBJS) tests/test_restrict_nl_3d.o -o $@ -lm

tests/test_prolong_nl_2d: $(OBJS) tests/test_prolong_nl_2d.o
	$(CC) -g $(OBJS) tests/test_prolong_nl_2d.o -o $@ -lm

tests/test_prolong_nl_3d: $(OBJS) tests/test_prolong_nl_3d.o
	$(CC) -g $(OBJS) tests/test_prolong_nl_3d.o -o $@ -lm

tests: $(TEST_PROGS) driver_multigrid
	cd tests && bash run_test.sh
	cd tests && bash run_test_child.sh
	cd tests && bash run_test_project.sh
	cd tests && bash run_test_prolong.sh
	cd tests && bash run_test_restrict_nl.sh
	cd tests && bash run_test_prolong_nl.sh
	cd tests && bash run_test_convergence.sh

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	$(RM) $(OBJS) \
	      driver_multigrid.o driver_multigrid \
	      parameter.o multigrid_parameters.o \
	      $(TEST_PROGS) $(TEST_OBJS)
	$(RM) -r tests/convergence_run

distclean: clean
	$(RM) *.json tests/*.json
