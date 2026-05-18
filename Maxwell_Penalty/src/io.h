#ifndef IO_H
#define IO_H
#include "gf.h"
#include "domain.h"

void output_gfs_metadata(struct ngfs *gfs);
void output_gfs_2D_xy(struct ngfs *gfs, const int global_k_index);
void output_gfs_2D_xy_h5(struct ngfs *gfs, const int global_k_index);
void output_gfs_3D_h5(struct ngfs *gfs);
void set_output_counter_3D(int value);
void set_output_counter_2D_xy_h5(int value);
void set_output_counter_2D_xy(int value);

void write_checkpoint(struct ngfs *gfs, double t, int it, int max_checkpoints);
int  read_checkpoint(struct ngfs *gfs, double *t, int *it);
#endif
