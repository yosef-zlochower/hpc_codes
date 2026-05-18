#!/usr/bin/env bash
# Verify that the committed simple_maxwell.h matches what generate_ccode.py
# would produce right now.  Catches stale generated code (hand-edits, or a
# SymPy script change without a corresponding regeneration).
#
# Usage: check_generated_simple_maxwell.sh <generator.py> <committed_header.h>
#
# Exits 0 on match, non-zero on mismatch.  Skips (exit 0) if the Python
# interpreter doesn't have SymPy installed, or if clang-format is missing —
# this lets the test run in CI on machines that have the toolchain and be
# a no-op on developer machines that don't.

set -u

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <generator.py> <committed_header.h>" >&2
    exit 2
fi

GENERATOR="$1"
COMMITTED="$2"

if [ ! -f "$GENERATOR" ];  then echo "generator not found: $GENERATOR" >&2; exit 2; fi
if [ ! -f "$COMMITTED" ];  then echo "header not found: $COMMITTED"  >&2; exit 2; fi

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "SKIP (no python3 on PATH; set PYTHON to override)"
    exit 0
fi

if ! "$PYTHON" -c "import sympy" >/dev/null 2>&1; then
    echo "SKIP (SymPy not installed in '$PYTHON'; set PYTHON to a venv that has it)"
    exit 0
fi

if ! command -v clang-format >/dev/null 2>&1; then
    echo "SKIP (clang-format not available; cannot normalise generator output)"
    exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cp "$GENERATOR" "$WORK/generate_ccode.py"
cp "$COMMITTED" "$WORK/committed.h"

# generate_ccode.py writes ./simple_maxwell.h in its cwd.
( cd "$WORK" && "$PYTHON" generate_ccode.py )

# Normalise both files with the same clang-format invocation so the diff
# reflects only semantic drift, not whitespace from the raw SymPy emitter.
clang-format -i "$WORK/simple_maxwell.h"
clang-format -i "$WORK/committed.h"

if diff -u "$WORK/committed.h" "$WORK/simple_maxwell.h" > "$WORK/diff.out"; then
    echo "ok (simple_maxwell.h matches generate_ccode.py output)"
    exit 0
else
    echo "FAIL: committed simple_maxwell.h does not match generate_ccode.py output."
    echo "Re-run:   python generate_ccode.py && clang-format -i simple_maxwell.h"
    echo "Diff (committed -> regenerated):"
    cat "$WORK/diff.out"
    exit 1
fi
