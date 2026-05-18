#ifndef IO_HPP
#define IO_HPP
#include "gf.hpp"

void output_gfs_metadata(NGFS *gfs);
void output_gfs_3D_h5  (NGFS *gfs);
void output_gfs_2D_xy_h5(NGFS *gfs, const int global_k_index);
void set_output_counter_3D(int value);
void set_output_counter_2D_xy_h5(int value);

void write_checkpoint(NGFS *gfs, double t, int it, int max_checkpoints);
int  read_checkpoint (NGFS *gfs, double *t, int *it);

#endif
