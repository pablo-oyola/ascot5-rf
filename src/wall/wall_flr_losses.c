#include "wall_flr_losses.h"
#include <stdlib.h>
#include <math.h>
#include "../consts.h"
#include "../gctransform.h"
#include "../B_field.h"
#include "../wall.h"
#include "../endcond.h"
#include "../error.h"

/**
 * @brief Evaluate finite Larmor radius (FLR) loss for a single GC marker.
 *
 * Samples a random gyrophase (uniform in [0,2pi)), performs first-order
 * guiding-center to particle transformation, and checks if the trajectory
 * segment from the guiding-center position to the instantaneous particle
 * position intersects the wall.
 *
 * @return 1 if wall was intersected (FLR loss), 0 otherwise.
 */
int flr_losses_eval(real r, real phi, real z, real ppar, real mu,
                    real mass, real charge, real time,
                    B_field_data* B, wall_data* wall, random_data* rnd,
                    int* walltile_out, int* err_out) {

    if(walltile_out) *walltile_out = 0;
    if(err_out) *err_out = 0;
 
    /* Sample random gyrophase */
    real zeta_rand;
    zeta_rand = random_uniform(rnd) * CONST_2PI;

    /* Evaluate magnetic field and gradients at GC position */
    real B_dB[15];
    a5err err = B_field_eval_B_dB(B_dB, r, phi, z, time, B);
    if(err) {
        if(err_out) *err_out = err;
        return 0; /* Cannot proceed */
    }

    /* Transform GC -> particle (first order) for sampled gyrophase */
    real rprt, phiprt, zprt;
    real pparprt, muprt, zetaprt; /* Unused outputs */
    gctransform_guidingcenter2particle(mass, charge, B_dB,
                                       r, phi, z, ppar, mu, zeta_rand,
                                       &rprt, &phiprt, &zprt,
                                       &pparprt, &muprt, &zetaprt);

    /* Check wall intersection */
    real w_coll = 0.0;
    int tile = wall_hit_wall(r, phi, z, rprt, phiprt, zprt, wall, &w_coll);
    if(tile > 0) {
        if(walltile_out) *walltile_out = tile;
        return 1;
    }
    return 0;
}