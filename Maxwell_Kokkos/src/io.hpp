#ifndef IO_HPP
#define IO_HPP
#include "gf.hpp"

void output_gfs_metadata(NGFS *gfs);
void output_gfs_3D_h5  (NGFS *gfs);
void set_output_counter_3D(int value);

void write_checkpoint(NGFS *gfs, double t, int it, int max_checkpoints);
int  read_checkpoint (NGFS *gfs, double *t, int *it);

#endif
