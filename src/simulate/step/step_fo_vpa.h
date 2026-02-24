/**
 * @file step_fo_vpa.h
 * @brief Header file for step_fo_vpa.c
 */
#ifndef STEP_FO_VPA_H
#define STEP_FO_VPA_H
#include "../../B_field.h"
#include "../../E_field.h"
#include "../../boozer.h"
#include "../../mhd.h"
#include "../../particle.h"
#include "../../icrh/RFlib.h"

// Setting the solver.
void set_push_function(int solver);

// 2nd order integrators, without phase correction.
void step_fo_vpa(particle_simd_fo* p, real* h, B_field_data* Bdata,
                 E_field_data* Edata, RF_fields* rffield_data);
void step_fo_vpa_mhd(particle_simd_fo* p, real* h, B_field_data* Bdata,
                     E_field_data* Edata, RF_fields* rffield_data, boozer_data* boozer,
                     mhd_data* mhd);

// With phase correction.
void step_fo_vpa_full(particle_simd_fo* p, real* h, B_field_data* Bdata,
                      E_field_data* Edata, RF_fields* rffield_data);
void step_fo_vpa_borisA(particle_simd_fo* p, real* h, B_field_data* Bdata,
                      E_field_data* Edata, RF_fields* rffield_data);

// Higher order integrators.
void step_fo_vpa_4th(particle_simd_fo* p, real* h, B_field_data* Bdata,
                     E_field_data* Edata, RF_fields* rffield_data);


// Defining the a type that describes a generic function to push the particles.
typedef void (*push_fo_fnt)(particle_simd_fo*, real*, B_field_data*, 
                            E_field_data*, RF_fields*);

// This is the function pointer that will be set to the desired solver.
// It defaults to the VPA phase corrected solver, which is the most 
// accurate one, and the one we want to use in our convergence tests. 
// The user can change it by setting the FULL_ORBIT_SOLVER option in the input file.
push_fo_fnt* full_orbit_pusher = step_fo_vpa_full;


#endif
