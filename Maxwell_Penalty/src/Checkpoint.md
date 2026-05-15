# Checkpoint and Recovery Plan for the Maxwell Solver

## 1. State Analysis

At the boundary between two timesteps (i.e., after `RK4_Step` returns and
`t += dt` has been applied), the complete simulation state consists of:

### 1.1 Minimal State Required for Recovery

| Item | Location in code | Size per rank | Notes |
|------|-----------------|---------------|-------|
| **Current time `t`** | local variable in `main()`, `driver.c:137` | 1 `double` | Scalar, same on all ranks |
| **Current iteration `it`** | loop counter in `main()`, `driver.c:182` | 1 `int` | Scalar, same on all ranks |
| **9 evolved variable arrays** (`Dx`, `Dy`, `Dz`, `Bx`, `By`, `Bz`, `PsiD`, `PsiB`, `rho`) | `gfs.vars[0..8]->new` | `9 * nx * ny * nz` doubles | These are the only arrays that carry forward information between timesteps. The `->old`, `->K1`..`->K4` arrays are scratch space overwritten at the start of each `RK4_Step`. |
| **3 material property arrays** (`ieps`, `imu`, `sigma`) | `gfs.auxvars[0..2]->new` | `3 * nx * ny * nz` doubles | Currently uniform but will become spatially varying when students implement inhomogeneous materials. Checkpointed to support that generality. |

### 1.2 State That Does NOT Need Checkpointing

| Item | Reason |
|------|--------|
| **`->old`, `->K1`, `->K2`, `->K3`, `->K4`** arrays | RK4 scratch; `->old` is copied from `->new` at the start of each step (`rk4.c:50-53`), and `K1`..`K4` are overwritten during the step. |
| **Constraint diagnostic arrays** (`cD`, `cB`) | Derived diagnostics recomputed before every output via `maxwell_constraints()`. |
| **Domain decomposition** (`domain3d_st`) | Fully reconstructed from the parameter file and MPI topology for the same process count. |
| **Communication buffers** | Transient; zeroed and repopulated during each `sync_vars` call. |
| **`maxwell_params`, `analytic_params`** | Re-read from the TOML parameter file at startup. `analytic_params` is derived from `maxwell_params` in `driver.c:87-95`. |
| **Static timer IDs** | Re-registered lazily on first use. |
| **Static output counters** (`io.c:57,100,156`) | These track HDF5 output group numbering. See discussion in Section 4.2. |
| **`l2_norm.dat` file** | Diagnostic output; can be regenerated or appended to. |

### 1.3 Key Insight

Because the time integrator is explicit (RK4) and self-starting, the state
between timesteps is **just the 9 evolved-variable arrays plus the scalar
time `t` (or equivalently the iteration count `it`, from which `t` can be
recomputed as `it * dt`), plus the 3 material property arrays**. The material
arrays are included because students will implement spatially varying fields.
Everything else is either re-derived from the parameter file or recomputed at
each step.

---

## 2. Checkpoint Design

### 2.1 When to Checkpoint

Checkpoint at the top of the time loop, after output (when `it % output_every == 0`),
or at a separate `checkpoint_every` interval controlled by a new TOML parameter.
This keeps the I/O phases together and ensures the data in `->new` is the
definitive state (not mid-RK4). A natural location is right after the existing
`output_gfs_3D_h5` call inside the `if (it % output_every == 0)` block:

```c
// driver.c, inside the time loop
if (it % checkpoint_every == 0)
{
    write_checkpoint(&gfs, t, it);
}
```

### 2.2 File Format

Use **one HDF5 file per MPI rank**, following the existing per-rank I/O pattern
(`3D_rank_N.h5`). Filename: `checkpoint_rank_N.h5`.

Each checkpoint file contains:

```
/metadata/                     (group, with attributes)
    ghost_zones    : int
    mpi_size       : int
    local_ni, local_nj, local_nk : int
    global_ni, global_nj, global_nk : int
    local_i0, local_j0, local_k0 : int
    dx, dy, dz     : double
    time            : double       <-- current simulation time
    iteration       : int          <-- current iteration number
    n_evol_vars     : int

/evolved/
    Dx   : 3D dataset [nk, nj, ni]
    Dy   : 3D dataset [nk, nj, ni]
    Dz   : 3D dataset [nk, nj, ni]
    Bx   : 3D dataset [nk, nj, ni]
    By   : 3D dataset [nk, nj, ni]
    Bz   : 3D dataset [nk, nj, ni]
    PsiD : 3D dataset [nk, nj, ni]
    PsiB : 3D dataset [nk, nj, ni]
    rho  : 3D dataset [nk, nj, ni]

/material/
    ieps  : 3D dataset [nk, nj, ni]   (1/epsilon)
    imu   : 3D dataset [nk, nj, ni]   (1/mu)
    sigma : 3D dataset [nk, nj, ni]   (conductivity)
```

The data written is `gfs->vars[v]->new` for each evolved variable, which is
the same data the existing `output_gfs_3D_h5` writes. The existing
`BinaryWriteArray` helper and `write_metadata` can be reused almost verbatim.

### 2.3 Atomicity

To avoid a corrupted checkpoint if the job is killed mid-write:

1. Write to temporary files: `checkpoint_rank_N.h5.tmp`
2. Call `MPI_Barrier` after all ranks finish writing.
3. Rename `.tmp` to `.h5` (rank 0 can do this, or each rank renames its own file).
4. Call `MPI_Barrier` again.

This ensures that either all checkpoint files are complete or none are.
Alternatively, rank 0 can write a small sentinel file `checkpoint.done` after
the barrier, and recovery checks for its existence.

### 2.4 Keeping Only the Latest Checkpoint

To limit disk usage, maintain at most two checkpoint sets: the current one
being written and the previous valid one. After the new checkpoint is confirmed
complete, delete the old one. This can be done by alternating between two
directory names (`checkpoint_A/`, `checkpoint_B/`) and a symlink
`checkpoint_latest -> checkpoint_A` that is atomically updated.

---

## 3. Recovery Design

### 3.1 Triggering Recovery

Add a boolean parameter to the TOML file:

```toml
[solver]
recover = false
checkpoint_every = 64
```

When `recover = true`, instead of calling `set_initial_data`, the driver reads
the checkpoint files and resumes.

### 3.2 Recovery Procedure

The recovery process (called in `main()` instead of `set_initial_data`) is:

```
1. Each rank opens  checkpoint_rank_<rank>.h5
2. Read /metadata/time       -> t
3. Read /metadata/iteration  -> it
4. Validate metadata against current run parameters:
     - global_ni, global_nj, global_nk must match
     - mpi_size must match (same number of MPI processes assumed)
     - ghost_zones must match
5. For v = 0..8:
     Read /evolved/<varname>  -> gfs.vars[v]->new
6. Read /material/ieps, /material/imu, /material/sigma
     -> gfs.auxvars[0..2]->new
7. Close file.
8. Resume the time loop starting at iteration  it + 1  with time  t.
```

### 3.3 Modified Driver Loop

The time loop in `driver.c` must be adjusted to start from the recovered
iteration rather than always from `it = 1`:

```c
int it_start = 1;
double t = 0.0;

if (maxwell_params.recover)
{
    read_checkpoint(&gfs, &t, &it_start);
    // Re-set material properties (they are not checkpointed)
    set_material_properties(&gfs);
}
else
{
    set_initial_data(&gfs, t);
}

// ... existing sync test, initial output, etc. (skip if recovering) ...

for (int it = it_start; it < maxwell_params.max_iterations; it++)
{
    RK4_Step(&gfs, t, dt, maxwell_eq_time_deriv);
    t += dt;
    // ... output and checkpoint logic ...
}
```

### 3.4 Handling the Static Output Counters

The functions `output_gfs_3D_h5`, `output_gfs_2D_xy`, and `output_gfs_2D_xy_h5`
each contain a `static int counter` that tracks the HDF5 group name (used as a
sequential output index). On recovery, these counters would restart from 0,
causing HDF5 group name collisions with pre-existing output files.

Options (in order of simplicity):
1. **Delete old output files before a recovery run.** Simplest; output is
   regenerated from the checkpoint forward.
2. **Initialize the counter from the checkpoint.** Store the output counter
   value in the checkpoint metadata. Add a function like
   `set_output_counter(int value)` to `io.c` that sets the static variable.
3. **Compute the counter from `it_start` and `output_every`.** Since
   `counter = it_start / output_every`, it can be derived without storing it.
   Add the same setter function and call it during recovery.

Option 3 is recommended: it requires no extra checkpoint data and is exact.

---

## 4. New/Modified Files Summary

| File | Change |
|------|--------|
| `parameter.h` / `maxwell_parameters.cc` | Add `recover` (bool) and `checkpoint_every` (int) fields to `maxwell_param_st`; parse them from TOML. |
| `maxwell.toml` | Add `recover = false` and `checkpoint_every = 64` under `[solver]`. |
| `io.h` / `io.c` | Add `write_checkpoint(struct ngfs *gfs, double t, int it)`, `read_checkpoint(struct ngfs *gfs, double *t, int *it)`, and `set_output_counter(int value)`. |
| `driver.c` | Add recovery branch before the time loop; add checkpoint call inside the time loop; adjust loop start from recovered iteration. |

No changes are needed to `rk4.c`, `gf.c`, `comm.c`, `domain.c`,
`maxwell_eqs.c`, or any of the physics/numerical code.

---

## 5. Implementation Sketch

### 5.1 `write_checkpoint`

```c
void write_checkpoint(struct ngfs *gfs, double t, int it)
{
    char filename[512], tmpname[512];
    snprintf(tmpname,  512, "checkpoint_rank_%d.h5.tmp", gfs->domain.rank);
    snprintf(filename, 512, "checkpoint_rank_%d.h5",     gfs->domain.rank);

    // Create HDF5 file, write metadata (reuse write_metadata pattern)
    // Add time and iteration as attributes in /metadata
    // Write each evolved variable under /evolved/<varname>
    // Close file

    MPI_Barrier(MPI_COMM_WORLD);
    rename(tmpname, filename);   // atomic on POSIX
    MPI_Barrier(MPI_COMM_WORLD);

    if (gfs->domain.rank == 0)
        fprintf(stderr, "Checkpoint written at t = %g, iteration %d\n", t, it);
}
```

### 5.2 `read_checkpoint`

```c
int read_checkpoint(struct ngfs *gfs, double *t, int *it)
{
    char filename[512];
    snprintf(filename, 512, "checkpoint_rank_%d.h5", gfs->domain.rank);

    // Open HDF5 file
    // Read /metadata attributes: validate global_ni/nj/nk, mpi_size, ghost_zones
    // Read time -> *t,  iteration -> *it
    // For v = 0..8: read /evolved/<varname> into gfs->vars[v]->new
    // Read /material/ieps, imu, sigma into gfs->auxvars[0..2]->new
    // Close file

    return 0;
}
```

---

## 6. Verification Strategy

To verify that checkpoint/recovery is implemented correctly:

1. **Bit-for-bit test**: Run the simulation for N steps, checkpoint at step M < N.
   Then run a fresh recovery from the checkpoint for (N - M) more steps. Compare
   the final output arrays between the two runs. They should be identical to
   machine precision (since RK4 is deterministic and the same MPI topology is
   used).

2. **L2 error continuity**: The `l2_norm.dat` error trajectory should be
   continuous across the checkpoint/recovery boundary with no jump.

3. **Constraint continuity**: The constraint violations `cD`, `cB` should show
   no discontinuity at the recovery point.
