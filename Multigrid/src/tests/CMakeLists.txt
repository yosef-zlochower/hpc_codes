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
    test_domain_2d test_domain_3d
    test_child_2d  test_child_3d
    test_project_2d test_project_3d
    test_prolong_2d test_prolong_3d
    test_restrict_nl_2d test_restrict_nl_3d
    test_prolong_nl_2d  test_prolong_nl_3d
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
    run_test_parser.sh
    run_test_convergence.sh
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
# manufactured_neumann_inhomog is registered in problem_registry.c
# but still NOT in this list -- now for a different reason than
# pre-CellCentred-Phase-2.  CellCentred Phase 2 fixed the
# compatibility issue (sum f = sum q/h matches the continuous
# condition exactly under cell-centred Neumann), and the V-cycle
# converges geometrically.  However, the simple ghost formula
# u_ghost = u_int + h*q has O(h^3) ghost truncation, which gives
# O(h) Laplacian truncation at the boundary cells when the exact
# solution has non-zero u'''(boundary).  For the homog preset
# (u = cos(pi x) cos(pi y) cos(pi z)) u'''(0) vanishes by symmetry,
# so the rate is a clean 2; for the inhomog preset
# (u = sin(pi x/2) sin(pi y/2) sin(pi z/2)) the rate stays at 2
# at coarse h but drifts toward 1.5 at h=1/128, just below the
# verifier's [1.8, 2.3] band.  CellCentred_plan.md Sec 4.2 already
# describes the 4-point higher-order ghost that fixes this; that's
# scheduled for Phase 5 (smoother retuning + boundary stencil).

set(VERIFY_SCRIPTS
    verify.py
    verify_zeros.py
    verify_nl_restrict.py
    verify_nl_prolong.py
    verify_convergence.py
    check.py
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
set(_e2e_scripts run_test_parser.sh)

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
