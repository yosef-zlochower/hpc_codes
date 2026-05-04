"""Generate simple_maxwell.h from a SymPy symbolic description of the
extended-Maxwell PDE.

The generated header defines four groups of macros, all consumed by
maxwell_eqs.c:

  SIMPLE_MAXWELL_DERIVS
      The spatial derivatives needed by the interior RHS, each named
      dx_X / dy_X / dz_X.  The caller supplies DIFFX/DIFFY/DIFFZ, so
      the same header works for both the deep-interior loop (fixed 4th-
      order central stencil) and the boundary-shell loop (runtime SBP
      closure).

  SIMPLE_MAXWELL_INTERIOR_DOT
      The time derivatives dotX[ijk] = ... for all nine evolved
      variables (D, B, PsiD, PsiB, rho).

  APPLY_SAT_LOWER_{X,Y,Z}(scale_, g_)
  APPLY_SAT_UPPER_{X,Y,Z}(scale_, g_)
      Six SBP-SAT penalty macros -- one per physical boundary face.
      Each subtracts  scale_ * A_face * (u - g_) from the dot fields
      at the boundary row, where A_face is the characteristic
      projection onto the incoming modes at that face.  A_face is
      derived from first principles: for each axis the principal-part
      flux matrix A_axis satisfies dt u + A_axis d_axis u = 0; we
      diagonalise it with SymPy, then take
          A_lower_face = (|A| - A)/2    -- projector onto A's positive
                                            eigenspace (incoming at lower)
          A_upper_face = (|A| + A)/2    -- projector onto A's negative
                                            eigenspace (incoming at upper)
      where |A| = P * |D| * P^{-1} comes straight from the eigen-
      decomposition A = P D P^{-1}.
"""

import re
import sys

import sympy as sp
from sympy import (Derivative, Function, IndexedBase, Idx, Matrix, Rational,
                   Symbol, Wild, ccode, zeros)
from sympy.vector import CoordSys3D, Del, curl, divergence


# ---------------------------------------------------------------------------
# Symbolic setup
# ---------------------------------------------------------------------------

Cart = CoordSys3D("Cart")
delop = Del()
x, y, z = Cart.x, Cart.y, Cart.z
nx_, ny_, nz_ = Cart.i, Cart.j, Cart.k

# Evolved fields (as SymPy Function(x,y,z) so that Del / curl / divergence work).
Dx_fn = Function("Dx")(x, y, z)
Dy_fn = Function("Dy")(x, y, z)
Dz_fn = Function("Dz")(x, y, z)

Bx_fn = Function("Bx")(x, y, z)
By_fn = Function("By")(x, y, z)
Bz_fn = Function("Bz")(x, y, z)

PsiB_fn = Function("PsiB")(x, y, z)
PsiD_fn = Function("PsiD")(x, y, z)

rho_fn = Function("rho")(x, y, z)

# Auxiliary (material) fields.
ieps_fn  = Function("ieps")(x, y, z)
imu_fn   = Function("imu")(x, y, z)
sigma_fn = Function("sigma")(x, y, z)

Dvec = Dx_fn * nx_ + Dy_fn * ny_ + Dz_fn * nz_
Bvec = Bx_fn * nx_ + By_fn * ny_ + Bz_fn * nz_
Hvec = Bvec * imu_fn
Evec = Dvec * ieps_fn
Jvec = sigma_fn * Evec

kappa_B = Symbol("kappa_B")
kappa_D = Symbol("kappa_D")
four_pi = Symbol("four_pi")

# Full interior evolution equations.
Bdot    = (-curl(Evec) + delop(PsiB_fn)).doit()
Ddot    = ( curl(Hvec) + delop(PsiD_fn) - four_pi * Jvec).doit()
PsiBdot = divergence(Bvec) - kappa_B * PsiB_fn
PsiDdot = divergence(Dvec) - four_pi * rho_fn - kappa_D * PsiD_fn
rhodot  = (-divergence(Jvec)).doit()

# Principal-part equations: same, but with the auxiliary fields frozen
# (treated as spatially varying *coefficients*, not evolved variables).
# These drive the characteristic analysis that defines the SAT penalties.
BBdot    = (-curl(Dvec) * ieps_fn + delop(PsiB_fn)).doit()
DBdot    = ( curl(Bvec) * imu_fn  + delop(PsiD_fn)).doit()
PsiBBdot = divergence(Bvec)
PsiDBdot = divergence(Dvec)


# ---------------------------------------------------------------------------
# Replacement rules:
#   Derivative(F, x) -> dx_F  (and analogous for y, z)
#   F(x,y,z)         -> F_scalar  (a Symbol), then wrap as IndexedBase(F)[ijk]
# IEPS, IMU are declared positive so SymPy knows sqrt(IEPS*IMU) is positive --
# this is important for the diagonalisation step further down.
# ---------------------------------------------------------------------------

coords = {x: "x", y: "y", z: "z"}
a, b = Wild("a"), Wild("b")
drule = Derivative(a, b), lambda a, b: Symbol("d" + coords[b] + "_" + a.name)

DX = Symbol("DX", real=True)
DY = Symbol("DY", real=True)
DZ = Symbol("DZ", real=True)
BX = Symbol("BX", real=True)
BY = Symbol("BY", real=True)
BZ = Symbol("BZ", real=True)
IEPS  = Symbol("IEPS",  positive=True)
IMU   = Symbol("IMU",   positive=True)
SIGMA = Symbol("SIGMA", real=True)
PSIB = Symbol("PSIB", real=True)
PSID = Symbol("PSID", real=True)
RHO  = Symbol("RHO",  real=True)

frule = {
    Dx_fn: DX, Dy_fn: DY, Dz_fn: DZ,
    Bx_fn: BX, By_fn: BY, Bz_fn: BZ,
    imu_fn: IMU, ieps_fn: IEPS, sigma_fn: SIGMA,
    PsiB_fn: PSIB, PsiD_fn: PSID, rho_fn: RHO,
}

Bdot    = Bdot.replace(*drule).subs(frule)
Ddot    = Ddot.replace(*drule).subs(frule)
rhodot  = rhodot.replace(*drule).subs(frule)
PsiBdot = PsiBdot.replace(*drule).subs(frule)
PsiDdot = PsiDdot.replace(*drule).subs(frule)

Bdotx, Bdoty, Bdotz = [Bdot.dot(n_) for n_ in (nx_, ny_, nz_)]
Ddotx, Ddoty, Ddotz = [Ddot.dot(n_) for n_ in (nx_, ny_, nz_)]

ijk = Idx("ijk", 1)
final_rule = {
    DX: IndexedBase("Dx")[ijk], DY: IndexedBase("Dy")[ijk], DZ: IndexedBase("Dz")[ijk],
    BX: IndexedBase("Bx")[ijk], BY: IndexedBase("By")[ijk], BZ: IndexedBase("Bz")[ijk],
    SIGMA: IndexedBase("sigma")[ijk], RHO: IndexedBase("rho")[ijk],
    PSIB:  IndexedBase("PsiB")[ijk], PSID:  IndexedBase("PsiD")[ijk],
    IMU:   IndexedBase("imu")[ijk],  IEPS:  IndexedBase("ieps")[ijk],
}

usedvar = set()
for ev in (Bdotx, Bdoty, Bdotz, Ddotx, Ddoty, Ddotz, rhodot, PsiBdot, PsiDdot):
    usedvar = usedvar.union(ev.free_symbols)

diffvar = [v for v in usedvar if re.match(r"^d[xyz]_", v.name)]
differentiated = sorted({v.name[3:] for v in diffvar})


# ---------------------------------------------------------------------------
# Characteristic analysis for the SAT penalties.
#
# The hyperbolic system has the form  dt u = M_a d_a u + lower-order  for
# each axis a in {x,y,z}.  We want the flux matrix A_a of the *standard*
# form dt u + A_a d_a u = 0, so A_a = -M_a.  At the lower-a face, the
# incoming characteristics are those with positive eigenvalue of A_a;
# at the upper-a face, negative eigenvalue of A_a.
#
# SAT penalty at lower face:  dot -= s * A_a^+ * (u - g)
#   where A_a^+ = (A_a + |A_a|) / 2 is the projector onto the positive-
#   eigenvalue eigenspace:  on each eigenmode of A_a with eigenvalue
#   lambda, A_a^+ acts as (lambda + |lambda|)/2 = max(lambda, 0).
# Since A_a = -M_a and |A_a| = |M_a|, this becomes
#     dot -= s * (|M_a| - M_a) / 2 * (u - g).
#
# Analogously at upper face:
#     dot -= s * (|M_a| + M_a) / 2 * (u - g).
#
# The factor-of-2 expansions of these expressions are what the hand-written
# APPLY_SAT_* macros do; we produce the same thing here by symbolic
# diagonalisation.
#
# Flux variables: 8-tuple, rho is excluded because its own flux is 0 (rho
# doesn't appear in any principal-part equation except its own, and there
# only through divergence(D) -- i.e. its row couples in from D's derivatives
# but its column contributes nothing).
# ---------------------------------------------------------------------------

funclist = (Bx_fn, By_fn, Bz_fn, Dx_fn, Dy_fn, Dz_fn, PsiB_fn, PsiD_fn)

allBdot = (
    BBdot.dot(nx_), BBdot.dot(ny_), BBdot.dot(nz_),
    DBdot.dot(nx_), DBdot.dot(ny_), DBdot.dot(nz_),
    PsiBBdot, PsiDBdot,
)


def _coeff_of_derivative(expr, func, direction):
    """Return the coefficient of Derivative(func, direction) in expr,
    evaluated with the scalar-replacement rules in frule."""
    marker = Symbol("_marker")
    term = Derivative(func, direction)
    # Substitute the target derivative with a marker, kill all other
    # direction-derivatives, then read off the coefficient.
    expr = expr.replace(term, marker)
    expr = expr.replace(Derivative(a, direction), 0)
    return expr.subs(marker, 1).subs(frule)


# derivs[direction][i] = the direction-derivative part of allBdot[i].
killdx = (Derivative(a, x), 0)
killdy = (Derivative(a, y), 0)
killdz = (Derivative(a, z), 0)
derivs = {
    x: [e.replace(*killdy).replace(*killdz) for e in allBdot],
    y: [e.replace(*killdx).replace(*killdz) for e in allBdot],
    z: [e.replace(*killdx).replace(*killdy) for e in allBdot],
}

# Build M_a for each axis:  M_a[i,j] = coefficient of d_a(u_j) in dt(u_i)
# (principal part).  The SAT flux matrix is A_a = -M_a.
M_axis = {}
for direction in (x, y, z):
    M = zeros(len(funclist), len(funclist))
    for i in range(len(funclist)):
        for j in range(len(funclist)):
            M[i, j] = _coeff_of_derivative(derivs[direction][i],
                                           funclist[j], direction)
    M_axis[direction] = M


def split_flux(M):
    """Return (A_lower, A_upper) where
        A_lower = (|M| - M)/2     # penalty projector at the lower face
        A_upper = (|M| + M)/2     # penalty projector at the upper face
    |M| is computed from the eigendecomposition M = P D P^{-1}.
    Eigenvalues are classified by substituting IEPS = IMU = 1 and checking
    the numeric sign; this is exact for the Maxwell flux matrix whose
    eigenvalues are always  +/-sqrt(IEPS*IMU), +/-1, or 0.
    """
    P, D = M.diagonalize()
    Pinv = P.inv()
    test_subs = {IEPS: 1, IMU: 1}
    lo_diag, hi_diag = [], []
    for lam in D.diagonal():
        sv = lam.subs(test_subs)
        if sv > 0:
            lo_diag.append(Rational(0))
            hi_diag.append(lam)
        elif sv < 0:
            lo_diag.append(-lam)          # |lam|
            hi_diag.append(Rational(0))
        else:
            lo_diag.append(Rational(0))
            hi_diag.append(Rational(0))
    A_lo = P * sp.diag(*lo_diag) * Pinv
    A_hi = P * sp.diag(*hi_diag) * Pinv
    return A_lo, A_hi


# Substitution rule used for emission: the SAT macros are written in
# terms of local scalars  _c = sqrt(ieps*imu)  and  _PP = sqrt(ieps/imu),
# so we rewrite  IEPS -> _c*_PP,  IMU -> _c/_PP  in the expressions.
_c_sym  = Symbol("_c",  positive=True)
_PP_sym = Symbol("_PP", positive=True)
c_pp_rule = {IEPS: _c_sym * _PP_sym, IMU: _c_sym / _PP_sym}

# Delta-variable names used in the emitted macros (match the hand-written
# style in maxwell_eqs.c: Psi fields shorten to PB / PD).
delta_name = {
    "Bx": "_dBx", "By": "_dBy", "Bz": "_dBz",
    "Dx": "_dDx", "Dy": "_dDy", "Dz": "_dDz",
    "PsiB": "_dPB", "PsiD": "_dPD",
}
delta_sym = {f.name: Symbol(delta_name[f.name]) for f in funclist}


# ---------------------------------------------------------------------------
# Emit simple_maxwell.h
# ---------------------------------------------------------------------------

def _print_continued(line):
    """Print one line of a multi-line #define, with a right-aligned
    backslash at column 76 for readability."""
    print(line.ljust(75) + "\\")


def emit_interior(*, Bdotx, Bdoty, Bdotz, Ddotx, Ddoty, Ddotz,
                  PsiBdot, PsiDdot, rhodot):
    """Print SIMPLE_MAXWELL_DERIVS + SIMPLE_MAXWELL_INTERIOR_DOT."""
    print("#define SIMPLE_MAXWELL_DERIVS ", end="")
    for name in differentiated:
        print("\\")
        print(f"    const double dx_{name} = DIFFX({name});\\")
        print(f"    const double dy_{name} = DIFFY({name});\\")
        print(f"    const double dz_{name} = DIFFZ({name});", end="")
    print()
    print()

    print("#define SIMPLE_MAXWELL_INTERIOR_DOT \\")
    print("    SIMPLE_MAXWELL_DERIVS;\\")

    def emit_row(lhs_name, rhs_expr, trailing=True):
        suffix = ";\\" if trailing else ";"
        print(f"    {IndexedBase(lhs_name)[ijk]} = "
              f"{ccode(rhs_expr.subs(final_rule))}{suffix}")

    emit_row("dotBx", Bdotx)
    emit_row("dotBy", Bdoty)
    emit_row("dotBz", Bdotz)
    print("    \\")
    emit_row("dotDx", Ddotx)
    emit_row("dotDy", Ddoty)
    emit_row("dotDz", Ddotz)
    print("    \\")
    emit_row("dotPsiB", PsiBdot)
    emit_row("dotPsiD", PsiDdot)
    emit_row("dotrho",  rhodot, trailing=False)
    print()


def emit_sat_macro(macro_name, A_matrix):
    """Print one APPLY_SAT_{LOWER,UPPER}_{X,Y,Z} macro."""
    _print_continued(f"#define {macro_name}(scale_, g_)")
    _print_continued("    do {")
    _print_continued("        const double _s  = (scale_);")
    _print_continued("        const double _PP = sqrt(ieps[ijk] / imu[ijk]);")
    _print_continued("        const double _c  = sqrt(ieps[ijk] * imu[ijk]);")
    # Delta declarations (all eight; even if a row's coefficient is zero
    # the delta is cheap to compute and keeps the macro uniform).
    for fld in funclist:
        fname = fld.name
        dname = delta_name[fname]
        _print_continued(
            f"        const double {dname} = {fname}[ijk] - (g_).{fname};"
        )
    # Per-field penalty updates.  Skip rows whose coefficient row is
    # identically zero in the current projection (e.g. rows that only
    # couple to the non-incoming eigenspace at this face).
    for i, fld in enumerate(funclist):
        fname = fld.name
        row_expr = sum(
            A_matrix[i, j] * delta_sym[funclist[j].name]
            for j in range(len(funclist))
        )
        row_expr = row_expr.subs(c_pp_rule)
        # Collect on _c so the EM rows come out as  (_c/2) * (linear combo
        # of deltas and _PP) rather than as a fully expanded sum.  For the
        # divergence rows, _c is absent and collect is a no-op.
        row_expr = sp.collect(sp.expand(row_expr), _c_sym)
        if row_expr == 0:
            continue
        code = ccode(row_expr)
        # SymPy emits Rational(1,2) as "1.0/2.0" (or "(1.0/2.0)" when used
        # as a prefix coefficient); normalise to 0.5 for pedagogical
        # clarity. The compiler emits identical machine code either way.
        code = code.replace("(1.0/2.0)", "0.5")
        code = code.replace("1.0/2.0",  "0.5")
        _print_continued(f"        dot{fname}[ijk] -= _s * ({code});")
    print("    } while (0)")
    print()


axis_name = {x: "X", y: "Y", z: "Z"}

saved_stdout = sys.stdout
with open("simple_maxwell.h", "w") as output_header:
    sys.stdout = output_header
    try:
        print("/* Machine-generated by generate_ccode.py.  Do not edit by hand. */")
        print()

        emit_interior(
            Bdotx=Bdotx, Bdoty=Bdoty, Bdotz=Bdotz,
            Ddotx=Ddotx, Ddoty=Ddoty, Ddotz=Ddotz,
            PsiBdot=PsiBdot, PsiDdot=PsiDdot, rhodot=rhodot,
        )

        for direction in (x, y, z):
            A_lo, A_hi = split_flux(M_axis[direction])
            emit_sat_macro(f"APPLY_SAT_LOWER_{axis_name[direction]}", A_lo)
            emit_sat_macro(f"APPLY_SAT_UPPER_{axis_name[direction]}", A_hi)
    finally:
        sys.stdout = saved_stdout
