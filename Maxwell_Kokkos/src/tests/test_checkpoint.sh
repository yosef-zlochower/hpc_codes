#!/bin/bash
# Test checkpoint/recovery by comparing a straight run against a
# checkpoint-then-recover run.  The final L2 errors must match
# bit-for-bit since RK4 is deterministic and we use the same
# MPI process count.
#
# Usage:  bash tests/test_checkpoint.sh [nprocs]
#   nprocs defaults to 2

set -e

NPROCS=${1:-2}
SRCDIR=$(cd "$(dirname "$0")/.." && pwd)
BINARY="$SRCDIR/maxwell_system"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run 'make' first." >&2
    exit 1
fi

# Use a small grid and short run for testing
TOTAL_ITERS=129   # run for 128 steps (loop goes 1..128)
CKPT_EVERY=64
OUTPUT_EVERY=64

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

echo "=== Checkpoint/recovery test (${NPROCS} MPI processes) ==="
echo "    work directory: $WORKDIR"

# Emit a parameter file in the current TOML schema.  Only the three
# run-control knobs vary between the reference / phase-1 / phase-2 runs,
# so the rest of the schema lives in exactly one place here — keeping the
# three configurations from drifting out of sync with the parser.
#
#   $1 = output TOML path
#   $2 = max_iterations
#   $3 = checkpoint_every   (0 disables checkpointing)
#   $4 = max_checkpoints
#   $5 = recover            (true/false)
#
# Configuration: the plane-wave source with periodic x,y and a physical
# z face is the documented non-dispersive convergence reference
# (Option B) — deterministic and cheap, exactly what a bit-for-bit
# checkpoint comparison needs.  Material is uniform so no lens sub-table
# is required.
write_toml() {
    cat > "$1" <<EOF
[grid]
nx = 16
ny = 16
nz = 16
x0 = 0.0
y0 = 0.0
z0 = 0.0
xn = 1.0
yn = 1.0
zn = 1.0
periodic_x = true
periodic_y = true
periodic_z = false

[solver]
ghost_size       = 3
cfl_factor       = 0.5
max_iterations   = $2
output_every     = $OUTPUT_EVERY
checkpoint_every = $3
max_checkpoints  = $4
recover          = $5
use_dissipation  = false
diss_coeff       = 0.1
tau              = 1.0

[physics]
kappa_D = 0.01
kappa_B = 0.01

[source]
type = "plane_wave"

[source.plane_wave]
ax     = 1.0
ay     = 1.0
k      = 2.0
bump_a = 0.0
bump_b = 2.0

[source.gaussian_beam]
w0        = 0.20
z_waist   = 0.5
k         = 15.0
amplitude = 1.0
ramp_a    = 0.0
ramp_b    = 1.0

[source.te_waveguide_mode]
l = 2
m = 2
n = 2

[material]
epsilon_type = "uniform"

[material.background]
epsilon = 1.0
mu      = 1.0
sigma   = 0.0
EOF
}

# ── Reference run: straight through, no checkpoint ──────────────────
REF_DIR="$WORKDIR/reference"
mkdir -p "$REF_DIR"
write_toml "$REF_DIR/maxwell.toml" "$TOTAL_ITERS" 0 1 false

echo "--- Reference run (straight through ${TOTAL_ITERS} iterations) ---"
cd "$REF_DIR"
mpirun --oversubscribe -np $NPROCS "$BINARY" maxwell.toml > stdout_ref.txt 2>stderr_ref.txt
echo "    done."

# ── Phase 1: run with checkpointing, stop at CKPT_EVERY ─────────────
CKPT_DIR="$WORKDIR/checkpoint"
mkdir -p "$CKPT_DIR"
# Run only to CKPT_EVERY+1 iterations (so iteration CKPT_EVERY completes and checkpoints)
PHASE1_ITERS=$((CKPT_EVERY + 1))
write_toml "$CKPT_DIR/maxwell.toml" "$PHASE1_ITERS" "$CKPT_EVERY" 2 false

echo "--- Phase 1: run to iteration $CKPT_EVERY with checkpointing ---"
cd "$CKPT_DIR"
mpirun --oversubscribe -np $NPROCS "$BINARY" maxwell.toml > stdout_p1.txt 2>stderr_p1.txt
echo "    done. Checking checkpoint files exist..."

for r in $(seq 0 $((NPROCS - 1))); do
    if ! ls checkpoint_it_*_rank_${r}.h5 >/dev/null 2>&1; then
        echo "FAIL: no checkpoint file found for rank ${r}!" >&2
        exit 1
    fi
done
echo "    checkpoint files present."

# ── Phase 2: recover and continue to TOTAL_ITERS ────────────────────
# Rewrite the TOML to enable recovery and run out to the full length.
write_toml "$CKPT_DIR/maxwell.toml" "$TOTAL_ITERS" 0 1 true

echo "--- Phase 2: recover from checkpoint and continue to iteration $((TOTAL_ITERS - 1)) ---"
cd "$CKPT_DIR"
mpirun --oversubscribe -np $NPROCS "$BINARY" maxwell.toml > stdout_p2.txt 2>stderr_p2.txt
echo "    done."

# ── Compare final L2 error values ──────────────────────────────────
# The reference l2_norm.dat has all entries from t=0.
# The checkpoint l2_norm.dat was opened in append mode for phase 2,
# so it should have entries from phase 1 and phase 2.
# We compare the LAST line of each file (the final L2 error).

REF_LAST=$(tail -1 "$REF_DIR/l2_norm.dat")
CKPT_LAST=$(tail -1 "$CKPT_DIR/l2_norm.dat")

echo ""
echo "Reference final L2:   $REF_LAST"
echo "Checkpoint final L2:  $CKPT_LAST"
echo ""

if [ "$REF_LAST" = "$CKPT_LAST" ]; then
    echo "PASS: Final L2 errors match bit-for-bit."
    exit 0
else
    echo "FAIL: Final L2 errors differ!" >&2
    echo "  Reference:  $REF_LAST" >&2
    echo "  Checkpoint: $CKPT_LAST" >&2
    exit 1
fi
