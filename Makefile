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

%.o: %.c  $(HDRS)
	mpicc  -c -Wall -O3 -ffast-math  -g  $<

tests/%.o: tests/%.c  $(HDRS)
	mpicc  -c -Wall -O3 -ffast-math  -g  -I.  -o $@  $<

gauss_seidel.o: gauss_seidel.c gauss_seidel.h $(HDRS)
	mpicc -c -Wall -O3 -ffast-math -g $<

multigrid.o: multigrid.c gauss_seidel.h $(HDRS)
	mpicc -c -Wall -O3 -ffast-math -g $<

driver.o: driver.c gauss_seidel.h $(HDRS)
	mpicc -c -Wall -O3 -ffast-math -g $<

driver_multigrid.o: driver_multigrid.c gauss_seidel.h multigrid.h $(HDRS)
	mpicc -c -Wall -O3 -ffast-math -g $<

driver: driver.o $(OBJS)
	mpicc -g $^ -o $@ -lm

driver_multigrid: driver_multigrid.o $(OBJS)
	mpicc -g $^ -o $@ -lm

all: $(OBJS)

tests/test_domain_2d: $(OBJS) tests/test_domain_2d.o
	mpicc -g $(OBJS)  tests/test_domain_2d.o -o $@ -lm

tests/test_domain_3d: $(OBJS) tests/test_domain_3d.o
	mpicc -g $(OBJS)  tests/test_domain_3d.o -o $@ -lm

tests/test_child_2d: $(OBJS) tests/test_child_2d.o
	mpicc -g $(OBJS) tests/test_child_2d.o -o $@ -lm

tests/test_child_3d: $(OBJS) tests/test_child_3d.o
	mpicc -g $(OBJS) tests/test_child_3d.o -o $@ -lm

tests/test_project_2d: $(OBJS) tests/test_project_2d.o
	mpicc -g $(OBJS) tests/test_project_2d.o -o $@ -lm

tests/test_project_3d: $(OBJS) tests/test_project_3d.o
	mpicc -g $(OBJS) tests/test_project_3d.o -o $@ -lm

tests/test_prolong_2d: $(OBJS) tests/test_prolong_2d.o
	mpicc -g $(OBJS) tests/test_prolong_2d.o -o $@ -lm

tests/test_prolong_3d: $(OBJS) tests/test_prolong_3d.o
	mpicc -g $(OBJS) tests/test_prolong_3d.o -o $@ -lm

tests/test_restrict_nl_2d: $(OBJS) tests/test_restrict_nl_2d.o
	mpicc -g $(OBJS) tests/test_restrict_nl_2d.o -o $@ -lm

tests/test_restrict_nl_3d: $(OBJS) tests/test_restrict_nl_3d.o
	mpicc -g $(OBJS) tests/test_restrict_nl_3d.o -o $@ -lm

tests/test_prolong_nl_2d: $(OBJS) tests/test_prolong_nl_2d.o
	mpicc -g $(OBJS) tests/test_prolong_nl_2d.o -o $@ -lm

tests/test_prolong_nl_3d: $(OBJS) tests/test_prolong_nl_3d.o
	mpicc -g $(OBJS) tests/test_prolong_nl_3d.o -o $@ -lm

tests: $(TEST_PROGS)
	cd tests && bash run_test.sh
	cd tests && bash run_test_child.sh
	cd tests && bash run_test_project.sh
	cd tests && bash run_test_prolong.sh
	cd tests && bash run_test_restrict_nl.sh
	cd tests && bash run_test_prolong_nl.sh

clean:
	-rm $(OBJS) driver.o driver driver_multigrid.o driver_multigrid $(TEST_PROGS) $(TEST_OBJS)
