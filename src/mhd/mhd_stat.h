/**
 * @file mhd_stat.h
 * @brief Header file for mhd_stat.c
 */
#ifndef MHD_STAT_H
#define MHD_STAT_H

#include "../ascot5.h"
#include "../offload.h"
#include "../error.h"
#include "../boozer.h"
#include "../spline/interp.h"
#include "../B_field.h"

/**
 * @brief MHD stat parameters on the target
 */
typedef struct {
    int n_modes;        /**< Number of modes            */
    real rho_min;       /**< rho grid minimum value     */
    real rho_max;       /**< rho grid maximum value     */
    int* nmode;         /**< Toroidal mode numbers      */
    int* mmode;         /**< Poloidal mode numbers      */
    real* amplitude_nm; /**< Amplitude of each mode     */
    real* omega_nm;     /**< Toroidal frequency [rad/s] */
    real* phase_nm;     /**< Phase of each mode [rad]   */

    /**
     * @brief 1D splines (rho) for each mode's magnetic eigenfunction
     */
    interp1D_data* alpha_nm;

    /**
     * @brief 1D splines (rho) for each mode's electric eigenfunction
     */
    interp1D_data* phi_nm;
    
    /**
     * @brief Mode group IDs for grouping modes that share amplitude/phase.
     *        If NULL, each mode is in its own group (index equals group ID).
     *        If set, modes with same group ID share the same amplitude/phase.
     */
    int* mode_group_id;
} mhd_stat_data;

int mhd_stat_init(mhd_stat_data* data, int nmode, int nrho,
                  real rhomin, real rhomax, int* moden, int* modem,
                  real* amplitude_nm, real* omega_nm, real* phase_nm,
                  real* alpha, real* phi);
void mhd_stat_free(mhd_stat_data* data);
void mhd_stat_offload(mhd_stat_data* data);
DECLARE_TARGET_SIMD_UNIFORM(boozerdata, mhddata, includemode, Bdata)
a5err mhd_stat_eval(
    real mhd_dmhd[10], real r, real phi, real z, real t, int includemode,
    boozer_data* boozerdata, mhd_stat_data* mhddata, B_field_data* Bdata);
DECLARE_TARGET_SIMD_UNIFORM(boozerdata, mhddata, Bdata, pertonly, includemode)
a5err mhd_stat_perturbations(
    real pert_field[7], real r, real phi, real z, real t, int pertonly,
    int includemode, boozer_data* boozerdata, mhd_stat_data* mhddata,
    B_field_data* Bdata);

DECLARE_TARGET_SIMD_UNIFORM(mhddata, mode)
a5err mhd_stat_eval_potentials(real *alpha, real *phi, real psi, int mode,
                               mhd_stat_data* mhddata);

DECLARE_TARGET_SIMD_UNIFORM(boozerdata, mhddata, Bdata, pertonly, includemode)
a5err mhd_stat_eval_perturbations_dt(real pert_field[14], real r, real phi, real z,
                                     real t, int pertonly, int includemode,
                                     boozer_data* boozerdata, mhd_stat_data* mhddata,
                                     B_field_data* Bdata);

void mhd_stat_update_log_amplitudes_phases(
    mhd_stat_data* data,
    real* dlnA_dt,
    real* dphi_dt,
    real dt,
    int* evolve_flags);

void mhd_stat_update_amplitudes_phases(
    mhd_stat_data* data,
    real* dA_dt,
    real* dphi_dt,
    real dt,
    int* evolve_flags);


#endif
