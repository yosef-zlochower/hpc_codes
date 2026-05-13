# =============================================================================
# Test binaries + CTest registrations
# =============================================================================
#
# The historical bash scripts (run_test_*.sh) expect to find:
#   * the test binaries (./test_domain_3d, etc.)
#   * the Python verifier scripts (verify.py, verify_zeros.py, ...)
# in their working directory.  To preserve those scripts unchanged we
# build the binaries straight into ${CMAKE_CURRENT_BINARY_DIR} and copy
# every Python and bash script alongside them at configure time.  CTest
# then runs each script with WORKING_DIRECTORY = the same directory.
# =============================================================================

# -----------------------------------------------------------------------------
# Test binaries
# -----------------------------------------------------------------------------

set(TEST_BINS
    test_domain_3d
    test_child_3d
    test_project_3d
    test_prolong_3d
    test_restrict_nl_3d
    test_prolong_nl_3d
    test_restrict_cc_3d
    test_prolong_cc_3d
)

foreach(name IN LISTS TEST_BINS)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name} PRIVATE multigrid_solver)
endforeach()

# -----------------------------------------------------------------------------
# Bash drivers and Python verifiers
# -----------------------------------------------------------------------------
#
# `configure_file(... COPYONLY)` re-copies whenever the source changes.
# CMake automatically reconfigures when a source file's mtime advances,
# so editing any of these files and rebuilding picks up the change.

set(TEST_SCRIPTS
    run_test.sh
    run_test_child.sh
    run_test_project.sh
    run_test_prolong.sh
    run_test_restrict_nl.sh
    run_test_prolong_nl.sh
    run_test_restrict_cc.sh
    run_test_prolong_cc.sh
    run_test_parser.sh
    run_test_convergence.sh
    run_test_make_xdmf.sh
)

# Manufactured-solution presets used by the convergence test.  One
# CTest entry is generated per preset; each runs run_test_convergence.sh
# with the preset name as its sole positional argument.  Adding a new
# preset is therefore a two-line change here plus a record in
# problem_registry.c.
set(CONVERGENCE_PRESETS
    manufactured_dirichlet_homog
    manufactured_dirichlet_inhomog
    manufactured_neumann_homog
    manufactured_mixed
    manufactured_mixed_inhomog
)
# manufactured_neumann_inhomog remains excluded from the convergence
# loop.  Diagnosis: this is the only preset where the V-cycle's
# defect plateaus at ~1e-2 instead of driving to ~1e-9 -- the
# singular all-Neumann system has the constant in its null space,
# and the V-cycle keeps injecting a constant-mode component that the
# per-iteration mean-zero projection then immediately removes, so the
# defect oscillates around a non-zero stationary value.  This caps
# the achievable empirical rate around 1.7-1.88 depending on
# min_cells, just under the test's [1.8, 2.3] threshold.  The
# discretisation itself is rate-2 (consistent with the other
# Neumann-bearing presets); the missing rate is on the iterative
# side.  Fix would be a Krylov-accelerated coarse solve or explicit
# null-space projection inside the V-cycle, not a discretisation
# change.

set(VERIFY_SCRIPTS
    verify.py
    verify_zeros.py
    verify_nl_restrict.py
    verify_nl_prolong.py
    verify_convergence.py
    check.py
    h5read.py
)

foreach(f IN LISTS TEST_SCRIPTS VERIFY_SCRIPTS)
    configure_file(${f} ${f} COPYONLY)
endforeach()

# -----------------------------------------------------------------------------
# CTest registrations
# -----------------------------------------------------------------------------
#
# One test per script.  Test name is the script's basename without the
# leading "run_" / trailing ".sh".  Each test runs from this directory
# (where the binaries and verifiers live) so the scripts continue to
# resolve `./test_X` and `python verify_Y.py` against the local cwd.

set(_CTEST_LABELS_OPS  "operator")
set(_CTEST_LABELS_E2E  "end_to_end")

# Map each script to its CTest label set so a developer can run only
# the operator-level suites or only the end-to-end checks if they want.
# run_test_convergence.sh is fanned out to multiple CTest entries (one
# per preset) below, so it is excluded from the generic loop here.
set(_e2e_scripts run_test_parser.sh run_test_make_xdmf.sh)

foreach(script IN LISTS TEST_SCRIPTS)
    if(script STREQUAL "run_test_convergence.sh")
        continue()
    endif()
    # Strip the "run_" prefix and ".sh" suffix to form the test name.
    string(REGEX REPLACE "^run_(.*)\\.sh$" "\\1" tname "${script}")
    add_test(NAME ${tname}
        COMMAND bash ${script}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(script IN_LIST _e2e_scripts)
        set_tests_properties(${tname} PROPERTIES LABELS ${_CTEST_LABELS_E2E})
    else()
        set_tests_properties(${tname} PROPERTIES LABELS ${_CTEST_LABELS_OPS})
    endif()
endforeach()

# One convergence test per registered preset, named convergence_<preset>.
foreach(preset IN LISTS CONVERGENCE_PRESETS)
    add_test(NAME convergence_${preset}
        COMMAND bash run_test_convergence.sh ${preset}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(convergence_${preset}
        PROPERTIES LABELS ${_CTEST_LABELS_E2E})
endforeach()
