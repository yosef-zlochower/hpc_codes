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
def default_homogeneous_dirichlet(_grid, y, _level):
    """
    Implements homogeneous Dirichlet Boundary Conditions
    """
    y[0] = 0
    y[-1] = 0


class Multigrid:
    """
    Class implementation for multigrid solver.
    """

    def __init__(
        self,
        num_xcells,
        grid=None,
        d0_coeff=None,
        d1_coeff=None,
        d2_coeff=None,
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
        self.d0_coeff = None
        self.src = None
        self.omega_sor = omega_sor
        self.omega_sor_standard = omega_sor
        self.boundary = boundary
        self.min_cells = min_cells
        self._iterations_required = None

        self.nx = self.num_xcells + 1

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
        if d0_coeff is not None:
            self.d0_coeff = d0_coeff.copy()
            assert len(d0_coeff) == num_xcells + 1
        else:
            self.d0_coeff = np.zeros(num_xcells + 1, dtype=np.float64)

        assert len(grid) == num_xcells + 1
        self.grid = grid.copy()

        if d1_coeff is not None:
            self.d1_coeff = d1_coeff.copy()
            assert len(d1_coeff) == num_xcells + 1
        else:
            self.d1_coeff = np.zeros(num_xcells + 1, dtype=np.float64)

        assert d2_coeff is not None
        assert len(d2_coeff) == num_xcells + 1
        self.d2_coeff = d2_coeff.copy()

        if src is not None:
            self.src = src.copy()
            assert len(src) == num_xcells + 1
        else:
            self.src=np.zeros(num_xcells+1, dtype=np.float64)


        self.h = grid[1] - grid[0]

        if initial_estimate is not None:
            assert type(initial_estimate) == np.ndarray
            assert initial_estimate.shape == grid.shape
            self.solution = initial_estimate.copy()
        else:
            self.solution = np.zeros_like(d2_coeff)

        self.boundary(grid, self.solution, self.level)

        self.child = None
        self.defect = np.zeros_like(grid)
        self.defect_linf = None

        self.gauss_seidel_smoothing = self.gauss_seidel_smoothing_2nd
        self.calculate_defect = self.calculate_defect_2nd

        if num_xcells % 2 == 0 and num_xcells // 2 >= min_cells:
            d2 = self.d2_coeff[::2].copy()
            d1 = self.d1_coeff[::2].copy()
            d0 = self.d0_coeff[::2].copy()
            g = self.grid[::2].copy()
            childsrc = np.zeros(num_xcells//2 + 1, dtype=np.float64)
            self.child = Multigrid(
                num_xcells // 2,
                grid=g,
                d0_coeff=d0,
                d1_coeff=d1,
                d2_coeff=d2,
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
        string = str(self.num_xcells)
        if self.child:
            string += " " + self.child.__str__()
        return string

    def __len__(self):
        return self.num_xcells + 1

    def gauss_seidel_smoothing_2nd(self, n_iters=DEFAULT_ITERATIONS, omega_override=None):
        dx = self.h
        sol = self.solution
        d0coef = self.d0_coeff
        d1coef = self.d1_coeff
        d2coef = self.d2_coeff
        src = self.src
        omega_sor = self.omega_sor
        grid = self.grid
        level = self.level

        if self._iterations_required is None:
            self._iterations_required = 0

        if omega_override is not None:
            omega_sor = omega_override

        if self.red_black_decouple:
            _gauss_seidel_smoothing_2_rb_ext(
                n_iters,
                len(sol),
                dx,
                grid,
                sol,
                d0coef,
                d1coef,
                d2coef,
                src,
                omega_sor,
                self.boundary,
                level,
            )
        else:
            _gauss_seidel_smoothing_2_ext(
                n_iters,
                len(sol),
                dx,
                grid,
                sol,
                d0coef,
                d1coef,
                d2coef,
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

        dx = self.h
        d0coef = self.d0_coeff
        d1coef = self.d1_coeff
        d2coef = self.d2_coeff
        src = self.src
        y = self.solution
        grid = self.grid

        defect = self.defect
        length = len(y)
        self.defect_linf = _calculate_defect_2_ext(
            length,
            dx,
            grid,
            y,
            d0coef,
            d1coef,
            d2coef,
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

        sol = self.solution
        for i in range(1, self.nx):
            self.parent.solution[i * 2] -= sol[i]
            self.parent.solution[i * 2 - 1] -= 0.5 * (sol[i - 1] + sol[i])

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
            for i in range(2, self.num_xcells, 2):
                self.child.src[i // 2] = self.defect[i]
        else:
            for i in range(2, self.num_xcells, 2):
                self.child.src[i // 2] = (
                    2 * self.defect[i] +
                    self.defect[i - 1] + self.defect[i + 1]
                ) / 4

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
    length,
    dx,
    grid,
    sol,
    d0coef,
    d1coef,
    d2coef,
    src,
    omega_sor,
    boundary,
    level,
):
    """
    Implements Gauss-Seidel smoothing with SOR.
    Performs `n_iters` iterations of Gauss-Seidel to solve
    d2coef(x) y'' + d1coef(x) * y' + d0coef(x) * y = src(x)
    on a uniform grid using second-order accurate finite differencing
    d0coef(x) < 0 is required, but not enforced.

    """

    dx2 = dx * dx

    # check if tolerance reached every 50 interations
    for _ in range(n_iters):
        boundary(grid, sol, level)

        for i in range(1, length - 1):
            fac2 = 1.0 / (2.0 * d2coef[i] - dx2 * d0coef[i])
            sol[i] = (1.0 - omega_sor) * sol[i] + omega_sor * (
                sol[i + 1] * (d2coef[i] + 0.5 * dx * d1coef[i])
                + sol[i - 1] * (d2coef[i] - 0.5 * dx * d1coef[i])
                - dx2 * src[i]
            ) * fac2
        boundary(grid, sol, level)


@njit
def _gauss_seidel_smoothing_2_rb_ext(
    n_iters,
    length,
    dx,
    grid,
    sol,
    d0coef,
    d1coef,
    d2coef,
    src,
    omega_sor,
    boundary,
    level,
):
    """
    Implements Gauss-Seidel smoothing with SOR.
    Performs `n_iters` iterations of Gauss-Seidel to solve
    d2coef(x) y'' + d1coef(x) * y' + d0coef(x) * y = src(x)
    on a uniform grid using second-order accurate finite differencing
    d0coef(x) < 0 is required, but not enforced.

    """

    dx2 = dx * dx

    # check if tolerance reached every 50 interations
    for _ in range(n_iters):
        boundary(grid, sol, level)

        for i in range(2, length - 1, 2):
            fac2 = 1.0 / (2.0 * d2coef[i] - dx2 * d0coef[i])
            sol[i] = (1.0 - omega_sor) * sol[i] + omega_sor * (
                sol[i + 1] * (d2coef[i] + 0.5 * dx * d1coef[i])
                + sol[i - 1] * (d2coef[i] - 0.5 * dx * d1coef[i])
                - dx2 * src[i]
            ) * fac2
        boundary(grid, sol, level)

        for i in range(1, length - 1, 2):
            fac2 = 1.0 / (2.0 * d2coef[i] - dx2 * d0coef[i])
            sol[i] = (1.0 - omega_sor) * sol[i] + omega_sor * (
                sol[i + 1] * (d2coef[i] + 0.5 * dx * d1coef[i])
                + sol[i - 1] * (d2coef[i] - 0.5 * dx * d1coef[i])
                - dx2 * src[i]
            ) * fac2
        boundary(grid, sol, level)


@njit
def _calculate_defect_2_ext(
    length,
    dx,
    grid,
    y,
    d0coef,
    d1coef,
    d2coef,
    src,
    defect,
):
    """
    Determines the error in `solution` (see description of `gauss_seidel_smoothing`)

    """

    idx = 1.0 / dx
    idx2 = idx * idx

    defect[0] = 0
    defect[-1] = 0

    for i in range(1, length - 1):
        defect[i] = (
            (y[i + 1] + y[i - 1] - 2 * y[i]) * idx2 * d2coef[i]
            + (y[i + 1] - y[i - 1]) * 0.5 * idx * d1coef[i]
            + y[i] * d0coef[i]
            - src[i]
        )

    return np.abs(defect).max()
