/**
 * @file simple_delta_f.h
 * @brief Spline interpolation library
 *
 * Simple delta-f method implementation assuming a constant energy, Pphi and mu gradient.
 * 
 * The particles' weights are considered as a deviation from the original distribution function
 * f_0.
 */

#ifndef SIMPLE_DELTA_F_H
#define SIMPLE_DELTA_F_H

#include "../ascot5.h"
#include "../offload.h"

typedef struct{
    real dfdE;     /**< Energy gradient of the distribution function */
    real dfdPphi;  /**< Pphi gradient of the distribution function  */
    real dfdmu;    /**< mu gradient of the distribution function    */
} simple_delta_f_data;

void simple_delta_f_init(simple_delta_f_data* str,
                         real dfdE,
                         real dfdPphi,
                         real dfdmu);

void simple_delta_f_eval(simple_delta_f_data* str,
                          real E0, real E,
                          real Pphi0, real Pphi,
                          real mu0, real mu,
                          real* delta_f);

#endif