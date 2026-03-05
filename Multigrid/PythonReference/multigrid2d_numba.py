"""
Module: Multigrid
Implements a simple 1-dimensional multigrid solver
"""

import numpy as np
import logging
from numba import njit

DEFAULT_MIN_CELLS = 100
DEFAULT_ITERATIONS = 100
DEFAULT_TOLERANCE = 1.0e-10
DEFAULT_MAX_VCYCLE_ITERS = 100


@njit
def default_homogeneous_dirichlet(_gridx, _gridy, y, _level):
    """
    Implements homogeneous Dirichlet Boundary Conditions
    """
    y[0, :] = 0
    y[-1, :] = 0
    y[:, 0] = 0
    y[:, -1] = 0


class Multigrid:
    """
    Class implementation for multigrid solver.
    """

    def __init__(
        self,
        num_xcells,
        num_ycells,
        gridx=None,
        gridy=None,
        src=None,
        omega_sor=1.5,
        parent=None,
        logger=None,
        boundary=default_homogeneous_dirichlet,
        min_cells=DEFAULT_MIN_CELLS,
        tol=DEFAULT_TOLERANCE,
        initial_estimate=None,
        red_black_decouple=False,
        direct_restriction=True,
        omega_high_error_limit=100,
        omega_low_error_limit=DEFAULT_TOLERANCE * 10,
    ):
        self.parent = parent
        self.num_xcells = num_xcells
        self.num_ycells = num_ycells
        self.src = None
        self.omega_sor = omega_sor
        self.boundary = boundary
        self.min_cells = min_cells
        self._iterations_required = None

        self.omega_sor_standard = omega_sor

        self._solution_found = False

        self.tol = tol
        self.red_black_decouple = red_black_decouple
        self.direct_restriction = direct_restriction

        self.omega_high_error_limit = omega_high_error_limit
        self.omega_low_error_limit = omega_low_error_limit

        if logger is None:
            logger = logging.getLogger("multigrid logger")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self.logger = logger
        if parent:
            self.level = self.parent.level + 1
        else:
            self.level = 0

        self.gridx = gridx.copy()
        self.gridy = gridy.copy()

        if src is not None:
            self.src = src.copy()
            assert src.shape == (num_xcells + 1, num_ycells + 1)

        assert gridx.shape == (num_xcells + 1, num_ycells + 1)
        assert gridy.shape == (num_xcells + 1, num_ycells + 1)

        self.hx = gridx[1, 0] - gridx[0, 0]
        self.hy = gridy[0, 1] - gridy[0, 0]

        self.nx = self.num_xcells + 1
        self.ny = self.num_ycells + 1

        if initial_estimate is not None:
            assert type(initial_estimate) == np.ndarray
            assert initial_estimate.shape == gridx.shape
            self.solution = initial_estimate.copy()
        else:
            self.solution = np.zeros_like(gridx)

        self.boundary(gridx, gridy, self.solution, self.level)

        self.child = None
        self.defect = np.zeros_like(gridx)
        self.defect_linf = None

        self.gauss_seidel_smoothing = self.gauss_seidel_smoothing_2nd
        self.calculate_defect = self.calculate_defect_2nd

        if (
            num_xcells % 2 == 0
            and num_xcells // 2 >= min_cells
            and num_ycells % 2 == 0
            and num_ycells // 2 >= min_cells
        ):
            gx = gridx[::2, ::2].copy()
            gy = gridy[::2, ::2].copy()
            childsrc = np.zeros((num_xcells//2 + 1, num_ycells//2 + 1), dtype=np.float64)
            self.child = Multigrid(
                num_xcells // 2,
                num_ycells // 2,
                gridx=gx,
                gridy=gy,
                src=childsrc,
                omega_sor=omega_sor,
                parent=self,
                logger=logger,
                boundary=boundary,
                min_cells=min_cells,
                tol=tol,
                omega_high_error_limit=omega_high_error_limit,
                omega_low_error_limit=omega_low_error_limit,
                red_black_decouple=red_black_decouple,
                direct_restriction=direct_restriction,
            )
        if parent:
            parent.child = self

    def __str__(self):
        string = str(self.num_xcells) + ", " + str(self.num_ycells)
        if self.child:
            string += " " + self.child.__str__()
        return string

    def gauss_seidel_smoothing_2nd(self, n_iters=DEFAULT_ITERATIONS, omega_override=None):
        dx = self.hx
        dy = self.hy
        sol = self.solution
        src = self.src
        omega_sor = self.omega_sor
        gridx = self.gridx
        gridy = self.gridy
        level = self.level

        if self._iterations_required is None:
            self._iterations_required = 0

        if omega_override is not None:
            omega_sor = omega_override

        if self.red_black_decouple:
            _gauss_seidel_smoothing_2_rb_ext(
                n_iters,
                self.nx,
                self.ny,
                dx,
                dy,
                gridx,
                gridy,
                sol,
                src,
                omega_sor,
                self.boundary,
                level,
            )
        else:
            _gauss_seidel_smoothing_2_ext(
                n_iters,
                self.nx,
                self.ny,
                dx,
                dy,
                gridx,
                gridy,
                sol,
                src,
                omega_sor,
                self.boundary,
                level,
            )
        self._iterations_required += n_iters

    def calculate_defect_2nd(self):
        """
        Determines the error in `solution` (see description of `gauss_seidel_smoothing`)

        """

        dx = self.hx
        dy = self.hy
        src = self.src
        y = self.solution
        gridx = self.gridx
        gridy = self.gridy

        defect = self.defect
        self.defect_linf = _calculate_defect_2_ext(
            self.nx,
            self.ny,
            dx,
            dy,
            gridx,
            gridy,
            y,
            src,
            defect,
        )

        if self.defect_linf > self.omega_high_error_limit:
            self.omega_sor = 1.0
        else:
            self.omega_sor = self.omega_sor_standard

        if self.defect_linf < self.omega_low_error_limit:
            self.omega_sor = 1.0


        if self.defect_linf < self.tol:
            self._solution_found = True
            self.omega_sor = self.omega_sor_standard

    def prolongate(self):
        """
        propagate solution to parent (i.e., correct parent)
        """
        if self.parent is None:
            return
        self.logger.debug(" * " * self.level + " " +
                          f"prolongate at {self.num_xcells}")

        _prolongate_ext(self.nx, self.ny, self.solution, self.parent.solution)



    def restrict(self):
        """
        Assign task to child
        """
        if self.child is None:
            return
        self.logger.debug(" * " * self.level + " " +
                          f"restrict at {self.num_xcells}")
        self.child.solution.fill(0)

        if self.direct_restriction:
            self.child.src[1:-1, 1:-1] = self.defect[2:-2:2, 2:-2:2]
        else:
            _restrict_ext(self.child.num_xcells, self.child.num_ycells,
                          self.child.src, self.defect)

    def vcycle(self, smoothing_iters=DEFAULT_ITERATIONS, subcycles=1):
        """
        Perform a single V-cycle
        """
        if self.parent is None:
            self.logger.debug("Starting Vcycle")
        self.gauss_seidel_smoothing(n_iters=smoothing_iters)
        self.calculate_defect()

        it = 0
        while self.child and self.defect_linf > self.tol and it < subcycles:
            self.restrict()
            self.child.vcycle(
                smoothing_iters=smoothing_iters,
                subcycles=subcycles,
            )
            self.gauss_seidel_smoothing(n_iters=smoothing_iters)
            self.gauss_seidel_smoothing(n_iters=smoothing_iters, omega_override=1.0)
            self.calculate_defect()
            it += 1

        if self.parent:
            self.prolongate()

        if self.parent is None:
            return self.defect_linf
        return None

    def solve_to_tolerance(
        self,
        smoothing_iters=DEFAULT_ITERATIONS,
        max_vcycle_iters=DEFAULT_MAX_VCYCLE_ITERS,
        subcycles=1,
    ):
        """
        Perform V-cyles until error is less than tolerance
        """
        self.calculate_defect()
        err = self.defect_linf
        it = 0
        self.logger.info(f"{err=} at {it=}")
        while err > self.tol:
            err = self.vcycle(
                smoothing_iters=smoothing_iters,
                subcycles=subcycles,
            )
            it += 1
            self.logger.info(f"{err=} at {it=}")
            if it > max_vcycle_iters and err > self.tol:
                raise ToleranceNotReached
        self.logger.info(f"solution achieved with tol={err}")


class ToleranceNotReached(Exception):
    """
    Raise when tolerance not reached
    """


@njit
def _gauss_seidel_smoothing_2_ext(
    n_iters,
    nx,
    ny,
    dx,
    dy,
    gridx,
    gridy,
    sol,
    src,
    omega_sor,
    boundary,
    level,
):
    """
    Implements Gauss-Seidel smoothing with SOR.
    Performs `n_iters` iterations of Gauss-Seidel to solve
    LAP u = src
    on a uniform grid using second-order accurate finite differencing

    """

    dx2 = dx * dx
    dy2 = dy * dy
    ifac = 1.0 / (2.0 * (dx2 + dy2))

    # check if tolerance reached every 50 interations
    for _ in range(n_iters):
        boundary(gridx, gridy, sol, level)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                sol[i, j] = (1.0 - omega_sor) * sol[i, j] + omega_sor * (
                    (sol[i + 1, j] + sol[i - 1, j]) * dy2
                    + (sol[i, j + 1] + sol[i, j - 1]) * dx2
                    - dx2 * dy2 * src[i, j]
                ) * ifac
        boundary(gridx, gridy, sol, level)


@njit
def _gauss_seidel_smoothing_2_rb_ext(
    n_iters,
    nx,
    ny,
    dx,
    dy,
    gridx,
    gridy,
    sol,
    src,
    omega_sor,
    boundary,
    level,
):
    """
    Implements Gauss-Seidel smoothing with SOR.
    Performs `n_iters` iterations of Gauss-Seidel to solve
    LAP u = src
    on a uniform grid using second-order accurate finite differencing

    """

    dx2 = dx * dx
    dy2 = dy * dy
    ifac = 1.0 / (2.0 * (dx2 + dy2))

    # check if tolerance reached every 50 interations
    for _ in range(n_iters):
        boundary(gridx, gridy, sol, level)

        for i in range(1, nx - 1):
            ieven = 1 - i % 2
            for j in range(1 + ieven, ny - 1, 2):
                sol[i, j] = (1.0 - omega_sor) * sol[i, j] + omega_sor * (
                    (sol[i + 1, j] + sol[i - 1, j]) * dy2
                    + (sol[i, j + 1] + sol[i, j - 1]) * dx2
                    - dx2 * dy2 * src[i, j]
                ) * ifac
        boundary(gridx, gridy, sol, level)

        for i in range(1, nx - 1):
            ieven = 1 - i % 2
            for j in range(2 - ieven, ny - 1, 2):
                sol[i, j] = (1.0 - omega_sor) * sol[i, j] + omega_sor * (
                    (sol[i + 1, j] + sol[i - 1, j]) * dy2
                    + (sol[i, j + 1] + sol[i, j - 1]) * dx2
                    - dx2 * dy2 * src[i, j]
                ) * ifac
        boundary(gridx, gridy, sol, level)


@njit
def _calculate_defect_2_ext(
    nx,
    ny,
    dx,
    dy,
    gridx,
    gridy,
    y,
    src,
    defect,
):
    """
    Determines the error in `solution` (see description of `gauss_seidel_smoothing`)

    """

    idx = 1.0 / dx
    idx2 = idx * idx

    idy = 1.0 / dy
    idy2 = idy * idy

    defect[0, :] = 0
    defect[-1, :] = 0
    defect[:, 0] = 0
    defect[:, -1] = 0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            defect[i, j] = (
                (y[i + 1, j] + y[i - 1, j] - 2 * y[i, j]) * idx2
                + (y[i, j + 1] + y[i, j - 1] - 2 * y[i, j]) * idy2
                - src[i, j]
            )

    return np.abs(defect).max()


@njit
def _prolongate_ext(self_nx, self_ny, self_solution, parent_solution):
    """
    propagate solution to parent (i.e., correct parent)
    """
    sol = self_solution
    for i in range(1, self_nx):
        for j in range(1, self_ny):

            # points common to both grids
            parent_solution[i * 2, j * 2] -= sol[i, j]

            # points on the same coordinate line (x)
            parent_solution[i * 2 - 1, j * 2] -= 0.5 * (sol[i - 1, j] + sol[i, j])

            # point on the same coordinate line (y)
            parent_solution[i*2, j * 2 - 1] -= 0.5 * (sol[i, j-1] + sol[i, j])

            # other
            parent_solution[i*2 -1, j * 2 - 1] -= 0.25 * (
                   sol[i-1, j -1] + sol[i-1, j] + sol[i, j-1] + sol[i, j])



@njit
def _restrict_ext(child_nxcells, child_nycells, child_src, defect):
    """
    Assign task to child
    """
    for i in range(1, child_nxcells):
        for j in range(1, child_nycells):
            fi, fj = 2*i, 2*j
            child_src[i, j] = (
            4 * defect[fi,   fj  ]
            + 2 * (  defect[fi+1, fj  ] + defect[fi-1, fj  ]
             + defect[fi,   fj+1] + defect[fi,   fj-1])
            + (    defect[fi+1, fj+1] + defect[fi+1, fj-1]
            + defect[fi-1, fj+1] + defect[fi-1, fj-1])
            ) / 16


