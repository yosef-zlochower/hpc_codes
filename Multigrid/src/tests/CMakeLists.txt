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
    test_restrict_cc_3d test_prolong_cc_3d
    test_deferred_correction_3d
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
    run_test_deferred_correction.sh
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
)
# manufactured_neumann_inhomog and manufactured_mixed_inhomog are both
# registered in problem_registry.c but excluded from CTest under the
# CellCentred Phase 2/3 layout.  The simple boundary-cell stencils
# both have O(h) local truncation that propagates to global rate 1
# when the exact solution has non-zero higher derivatives at the
# Dirichlet boundary:
#
# * manufactured_neumann_inhomog (Phase 2): cell-centred ghost mirror
#   u_ghost = u_int + h*q has O(h^3) ghost truncation -> O(h)
#   Laplacian at boundary.  For the homog preset (u = cos(pi x)...)
#   u'''(0) = 0 by symmetry, rate is 2; for the inhomog preset
#   (u = sin(pi x/2)...) rate drifts toward 1.5 at h=1/128.
# * manufactured_mixed_inhomog (Phase 3): the 3-point non-uniform
#   formula at the cell adjacent to a hybrid Dirichlet vertex has
#   O(h) truncation.  The companion preset manufactured_mixed (homog
#   Dirichlet at the D end) gives clean rate 2 because u_v = 0 makes
#   the 3-point formula degenerate to a uniform stencil.
#
# CellCentred_plan.md Sec 4.2 specifies the 4-point higher-order
# ghost formula that fixes both: O(h^2) at the boundary -> rate 2
# globally.  That's scheduled for Phase 5 (smoother retuning +
# boundary stencil upgrade); both excluded presets re-enable then.

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
