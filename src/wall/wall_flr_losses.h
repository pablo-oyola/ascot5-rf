/**
 * @file flr_losses.h
 * @brief Header file for flr_losses.c. Contains the routines to 
 * determine the FLR losses of GC markers.
 */
#ifndef FLR_LOSSES_H
#define FLR_LOSSES_H

#include "../ascot5.h"
#include "../offload.h"
#include "../B_field.h"
#include "../wall.h"
#include "../random.h"
#include "../error.h"

/**
 * @brief Evaluate FLR wall loss for a single guiding center marker.
 *
 * Samples a random gyrophase, performs first-order guiding-center to particle
 * transform, and checks if the line segment from GC position to particle
 * position intersects the wall. Does not mutate global particle SIMD arrays;
 * instead returns intersection info through output pointers.
 *
 * @param r      Guiding center R [m]
 * @param phi    Guiding center phi [rad]
 * @param z      Guiding center Z [m]
 * @param ppar   Parallel momentum [kg m/s]
 * @param mu     Magnetic moment [J/T]
 * @param mass   Particle mass [kg]
 * @param charge Particle charge [C]
 * @param time   Simulation time [s] (for time-dependent B)
 * @param B      Magnetic field data
 * @param wall   Wall data
 * @param rnd    Random generator state.
 * @param walltile_out  (output) Wall element ID if lost (0 otherwise)
 * @param err_out       (output) Error flag (set if B-field eval fails)
 * @return 1 if FLR loss occurred, 0 otherwise
 */
GPU_DECLARE_TARGET_SIMD_UNIFORM(B, wall, rnd)
int flr_losses_eval(real r, real phi, real z, real ppar, real mu,
					real mass, real charge, real time,
					B_field_data* B, wall_data* wall, random_data* rnd,
					int* walltile_out, int* err_out);
GPU_DECLARE_TARGET_SIMD_UNIFORM_END

#endif 