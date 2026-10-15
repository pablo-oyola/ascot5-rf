#include "diag_energy_exchange.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "../B_field.h"
#include "../consts.h"
#include "../physlib.h"
#include "../print.h"
#include "../plasma.h"
#include "../mhd/mhd_stat.h"

#define PSI_POINTS 1024 /**< Number of psi points for interpolation. */
#ifndef M_PI
#define M_PI 3.14159265358979323846 /**< Definition of pi if not defined. */
#endif

/* Full-orbit field cache. Per mode and marker it holds the single-mode
 * field E (3) and dE/dt (3) at the end point of the marker's last step, and
 * the time of that point. In the fixed-step loop the next step starts from
 * an exact copy of that state, so its start-point field is reused instead
 * of evaluated again; an entry is used only if its time equals the start
 * time of the current step, so a marker's first step, and any step that does
 * not continue the previous one, evaluates the start point afresh. */
#define FO_CACHE_STRIDE 7
#define FO_CACHE_TIME   6
#define FO_CACHE_EMPTY  (-DBL_MAX)

/** @brief Mark every cache entry as empty (no-op without a cache). */
static void fo_cache_reset(diag_energy_exchange_data* data) {
    if(data->cached_fields_i == NULL) return;
    size_t n = (size_t)data->n_modes * (size_t)data->nprt;
    for(size_t k = 0; k < n; k++) {
        data->cached_fields_i[k * FO_CACHE_STRIDE + FO_CACHE_TIME] = FO_CACHE_EMPTY;
    }
}

/** @brief Allocate and empty the cache for the current size. */
static void fo_cache_allocate(diag_energy_exchange_data* data) {
    size_t n = (size_t)data->n_modes * (size_t)data->nprt * FO_CACHE_STRIDE;
    data->cached_fields_i = (real*)malloc(n * sizeof(real));
    fo_cache_reset(data);
}

real simpson(real* f, int n, real dx) {
    real sum = f[0] + f[n-1];
    for (int i = 1; i < n-1; i++) {
        sum += (i % 2 == 0 ? 2 : 4) * f[i];
    }
    return sum * dx / 3.0;
}

/**
 * @brief Evaluate unit-amplitude MHD eigenfunctions at a particle position.
 *
 * The interpolated alpha and phi profiles are returned before applying the
 * mode phase or amplitude. Theta and zeta are returned so the caller can
 * evaluate the phase separately. All outputs remain zero when the point is
 * outside the Boozer/MHD grid or interpolation fails.
 */
DECLARE_TARGET_SIMD_UNIFORM(boozerdata, mhddata, Bdata, mode)
static a5err diag_energy_exchange_eval_unscaled_potentials(
    real* alpha0, real* phipot0, real* theta, real* zeta,
    real r, real phi, real z, int mode, boozer_data* boozerdata,
    mhd_stat_data* mhddata, B_field_data* Bdata) {

    a5err err = 0;
    real ptz[12];
    int isinside = 0;

    *alpha0 = 0.0;
    *phipot0 = 0.0;
    *theta = 0.0;
    *zeta = 0.0;

    err = boozer_eval_psithetazeta(ptz, &isinside, r, phi, z, Bdata,
                                   boozerdata);
    if(err || !isinside) {
        return err;
    }

    real rho[2];
    err = B_field_eval_rho(rho, ptz[0], Bdata);
    if(err) {
        return err;
    }

    real alpha_eval = 0.0;
    real phipot_eval = 0.0;
    int interperr = 0;
    interperr += interp1Dcomp_eval_f(
        &alpha_eval, &(mhddata->alpha_nm[mode]), rho[0]);
    interperr += interp1Dcomp_eval_f(
        &phipot_eval, &(mhddata->phi_nm[mode]), rho[0]);
    if(interperr) {
        return err;
    }

    *alpha0 = alpha_eval;
    *phipot0 = phipot_eval;
    *theta = ptz[4];
    *zeta = ptz[8];
    return err;
}

/** @brief Evaluate the sine and cosine of one MHD mode phase. */
DECLARE_TARGET_SIMD_UNIFORM(mhddata, mode)
static void diag_energy_exchange_eval_mode_phase(
    real* sinmhd, real* cosmhd, real theta, real zeta, real t, int mode,
    mhd_stat_data* mhddata) {

    real mhdarg = mhddata->nmode[mode] * zeta
        - mhddata->mmode[mode] * theta
        - mhddata->omega_nm[mode] * t
        + mhddata->phase_nm[mode];
    *sinmhd = sin(mhdarg);
    *cosmhd = cos(mhdarg);
}

/**
 * @brief Initialize the energy exchange diagnostic data structure.
 *
 * This function initializes the diagnostic data structure for energy exchange
 * calculations, allocating memory for source terms and energy change.
 *
 * @param data Pointer to the diagnostic data structure to be initialized.
 * @param nprt Number of particles.
 * @param boozerdata Pointer to Boozer coordinate data.
 * @param B_data Pointer to magnetic field data.
 * @param mhd_data Pointer to MHD mode data.
 */
void diag_energy_exchange_init(diag_energy_exchange_data* data, 
                               int nprt, boozer_data* boozerdata,
                               B_field_data* B_data,
                               mhd_stat_data* mhd_data, 
                               plasma_data* plasma_data) {

    if (data == NULL || boozerdata == NULL || \
        B_data == NULL || mhd_data == NULL  || plasma_data == NULL) {
        print_err("Error: Null pointer passed to diag_energy_exchange_init.\n");
        return;
    }

    if(data->enabled == 1) diag_energy_exchange_free(data);

    data->n_modes = mhd_data->n_modes;
    data->nprt = nprt;
    data->mhd_data = mhd_data;
    data->boozerdata = boozerdata;
    data->B_data = B_data;
    data->plasma_data = plasma_data;

    // The energy exchange diagnostic, that can be used for understanding
    // mode evolution requires that the MHD has a non-zero frequency. 
    // We will set here a flag indicating which modes require this mode evolution.
    if (data->n_modes <= 0) {
        print_err("Error: No MHD modes defined.\n");
        return;
    }
    data->is_mode_evol = (int*)malloc(data->n_modes * sizeof(int));

    for (int i = 0; i < data->n_modes; i++) {
        if (mhd_data->omega_nm[i] == 0.0) {
            data->is_mode_evol[i] = 0; // No evolution for this mode
        } else {
            data->is_mode_evol[i] = 1; // Evolution is required
        }
    }

    // Allocate memory for source terms and energy change
    data->S1 = (real*)malloc(data->n_modes * nprt * sizeof(real));
    data->S2 = (real*)malloc(data->n_modes * nprt * sizeof(real));
    data->dEnergy = (real*)malloc(data->n_modes * nprt * sizeof(real));
    fo_cache_allocate(data);
    memset(data->S1, 0, data->n_modes * nprt * sizeof(real));
    memset(data->S2, 0, data->n_modes * nprt * sizeof(real));
    memset(data->dEnergy, 0, data->n_modes * nprt * sizeof(real));

    // Finally we get from the plasma data the mass of the main ion species.
    data->thrmass = plasma_get_species_mass(plasma_data)[1];

    data->enabled = 1; // Set enabled flag
}

/**
 * @brief Free allocated resources for the energy exchange diagnostic data.
 *
 * This function frees the memory allocated for the diagnostic data structure.
 *
 * @param data Pointer to the diagnostic data structure to be freed.
 */
void diag_energy_exchange_free(diag_energy_exchange_data* data) {
    if(data == NULL) return;
    if(data->enabled == 0) return;
    if (data->S1) free(data->S1);
    if (data->S2) free(data->S2);
    if (data->dEnergy) free(data->dEnergy);
    if (data->is_mode_evol) free(data->is_mode_evol);
    if (data->cached_fields_i) free(data->cached_fields_i);
    data->cached_fields_i = NULL;
    data->mhd_data = NULL;
    data->boozerdata = NULL;
    data->n_modes = 0;
    data->nprt = 0;

    data->enabled = 0; // Reset enabled flag
}

/**
 * @brief Offload data to the accelerator.
 *
 * This function prepares the diagnostic data for offloading to an accelerator.
 * It is a placeholder for future implementation.
 *
 * @param data Pointer to the diagnostic data structure.
 */
void diag_energy_exchange_offload(diag_energy_exchange_data* data) {
    GPU_MAP_TO_DEVICE(data->S1[0: data->n_modes * data->nprt], 
                    data->S2[0: data->n_modes * data->nprt], 
                    data->dEnergy[0: data->n_modes * data->nprt]);
    if(data->cached_fields_i != NULL) {
        GPU_MAP_TO_DEVICE(
            data->cached_fields_i[0: data->n_modes * data->nprt * FO_CACHE_STRIDE]);
    }
}


void diag_energy_exchange_onload(diag_energy_exchange_data* data) {
    GPU_MAP_FROM_DEVICE(data->S1[0: data->n_modes * data->nprt], 
                        data->S2[0: data->n_modes * data->nprt], 
                        data->dEnergy[0: data->n_modes * data->nprt]);

}



/**
 * @brief Energy exchange calculation for full-orbit particles.
 *
 * Accumulates, with the trapezoidal rule over each orbit step, the same three
 * quantities as the guiding-centre version:
 *
 * - dEnergy: the work done by the mode, int q v.E dt (amplitude included);
 * - S1: the unit-amplitude in-phase power, sum w int mu0 vA^2 q v.E/A dt;
 * - S2: the unit-amplitude quadrature power, sum w int mu0 vA^2 q v.E_q/A dt,
 *   where E_q = -(1/omega) dE/dt is the field of the same mode advanced by a
 *   quarter period in phase.
 *
 * The unit-amplitude field is the single-mode perturbation divided by that
 * mode's current amplitude, so S1 and S2 do not depend on the amplitude
 * explicitly. Velocities are p/(gamma m).
 *
 * @param data Pointer to the diagnostic data structure.
 * @param p_f Pointer to the full-orbit particle at the end of the time step.
 * @param p_i Pointer to the full-orbit particle at the start of the time step.
 */
void diag_energy_exchange_update_fo(diag_energy_exchange_data* data,
                                    particle_simd_fo* p_f,
                                    particle_simd_fo* p_i){

    GPU_PARALLEL_LOOP_ALL_LEVELS
    for(int i = 0; i < p_f->n_mrk; i++) {
        if(p_i->running[i] == 0) continue;
        int index = p_f->index[i];

        real dt = p_f->time[i] - p_i->time[i];
        real charge = p_i->charge[i];
        real mass = p_i->mass[i];
        real w0 = p_i->weight[i];
        real w1 = p_f->weight[i];

        /* Relativistic velocities from the momenta. */
        real pnorm_f = sqrt(p_f->p_r[i] * p_f->p_r[i]
                            + p_f->p_phi[i] * p_f->p_phi[i]
                            + p_f->p_z[i] * p_f->p_z[i]);
        real pnorm_i = sqrt(p_i->p_r[i] * p_i->p_r[i]
                            + p_i->p_phi[i] * p_i->p_phi[i]
                            + p_i->p_z[i] * p_i->p_z[i]);
        real ginv_f = 1.0 / (physlib_gamma_pnorm(mass, pnorm_f) * mass);
        real ginv_i = 1.0 / (physlib_gamma_pnorm(mass, pnorm_i) * mass);
        real v_f[3] = {p_f->p_r[i] * ginv_f, p_f->p_phi[i] * ginv_f,
                       p_f->p_z[i] * ginv_f};
        real v_i[3] = {p_i->p_r[i] * ginv_i, p_i->p_phi[i] * ginv_i,
                       p_i->p_z[i] * ginv_i};

        /* mu0 vA^2 = B^2 / (n_i m_i) at both endpoints. */
        real iondens_f, iondens_i;
        plasma_eval_dens(&iondens_f, p_f->rho[i],
            p_f->r[i], p_f->phi[i], p_f->z[i], p_f->time[i], 1, data->plasma_data);
        plasma_eval_dens(&iondens_i, p_i->rho[i],
            p_i->r[i], p_i->phi[i], p_i->z[i], p_i->time[i], 1, data->plasma_data);
        real B[3];
        B_field_eval_B(B, p_f->r[i], p_f->phi[i], p_f->z[i], p_f->time[i], data->B_data);
        real vA2_f = (B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
                   / (iondens_f * data->thrmass);
        B_field_eval_B(B, p_i->r[i], p_i->phi[i], p_i->z[i], p_i->time[i], data->B_data);
        real vA2_i = (B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
                   / (iondens_i * data->thrmass);

        for (int j = 0; j < data->n_modes; j++) {
            if( data->is_mode_evol[j] == 0 ) continue;
            real amplitude = data->mhd_data->amplitude_nm[j];
            real omega = data->mhd_data->omega_nm[j];
            if( amplitude == 0.0 ) continue; /* no unit field to recover */

            /* A failed evaluation (e.g. outside the B or Boozer domain)
             * counts as zero field, as it does for the pusher. */
            real pert[14];
            real E_f[3], dE_f[3], E_i[3], dE_i[3];
            if(mhd_stat_eval_perturbations_dt(pert, p_f->r[i], p_f->phi[i],
                   p_f->z[i], p_f->time[i], 1, j, data->boozerdata,
                   data->mhd_data, data->B_data)) {
                for(int k = 0; k < 14; k++) pert[k] = 0.0;
            }
            for(int k = 0; k < 3; k++) {
                E_f[k] = pert[3 + k];
                dE_f[k] = pert[10 + k];
            }

            /* Start point: the cached end point of the previous step if it
             * is this step's start time, otherwise a fresh evaluation. */
            real* cache = data->cached_fields_i == NULL ? NULL
                : &data->cached_fields_i[((size_t)j * (size_t)data->nprt
                                          + (size_t)index) * FO_CACHE_STRIDE];
            if(cache != NULL && cache[FO_CACHE_TIME] == p_i->time[i]) {
                for(int k = 0; k < 3; k++) {
                    E_i[k] = cache[k];
                    dE_i[k] = cache[3 + k];
                }
            }
            else {
                if(mhd_stat_eval_perturbations_dt(pert, p_i->r[i], p_i->phi[i],
                       p_i->z[i], p_i->time[i], 1, j, data->boozerdata,
                       data->mhd_data, data->B_data)) {
                    for(int k = 0; k < 14; k++) pert[k] = 0.0;
                }
                for(int k = 0; k < 3; k++) {
                    E_i[k] = pert[3 + k];
                    dE_i[k] = pert[10 + k];
                }
            }
            if(cache != NULL) {
                for(int k = 0; k < 3; k++) {
                    cache[k] = E_f[k];
                    cache[3 + k] = dE_f[k];
                }
                cache[FO_CACHE_TIME] = p_f->time[i];
            }

            real Edotv_f = E_f[0] * v_f[0] + E_f[1] * v_f[1] + E_f[2] * v_f[2];
            real Edotv_i = E_i[0] * v_i[0] + E_i[1] * v_i[1] + E_i[2] * v_i[2];
            real dEdotv_f = dE_f[0] * v_f[0] + dE_f[1] * v_f[1] + dE_f[2] * v_f[2];
            real dEdotv_i = dE_i[0] * v_i[0] + dE_i[1] * v_i[1] + dE_i[2] * v_i[2];

            /* 1. Work done by the mode (physical amplitude). */
            data->dEnergy[j * data->nprt + index] +=
                0.5 * charge * (Edotv_f + Edotv_i) * dt;

            /* 2. Unit-amplitude in-phase and quadrature powers. */
            real scale = charge / amplitude;
            real P_f = scale * Edotv_f;
            real P_i = scale * Edotv_i;
            real Q_f = -scale * dEdotv_f / omega;
            real Q_i = -scale * dEdotv_i / omega;
            data->S1[j * data->nprt + index] +=
                0.5 * (vA2_f * P_f * w1 + vA2_i * P_i * w0) * dt;
            data->S2[j * data->nprt + index] +=
                0.5 * (vA2_f * Q_f * w1 + vA2_i * Q_i * w0) * dt;
        }
    }

}


/**
* @brief Energy exchange calculation for guiding center particles.
*
* This function updates the energy exchange diagnostic data based on the
* interaction between a guiding center particle and the MHD modes, based on the
* end and start conditions of the particle for a given time step.
*
* Since the total energy difference in the p_i and p_f may also include other
* factors, such as drag and diffusion, RF, etc and it is the total energy
* difference, not discriminating by mode numbers (n, m), this function will
* give instead a comprehensive diagnostic on the energy exchange.
*
* To compute the energy exchange and the current coupling terms, we will use
* the trapezoidal rule to integrate over the time step of the particle.
*
* The version of the energy exchange here deployed is the one from the book
* R. B. White, "The theory of toroidally confined plasmas", 3rd edition,
* Oxford University Press, 2014, section 6.9.
*
* @todo This should be done more efficiently, there is no need to evaluate again
* the initial and final electric fields: just the final one is enough.
* @param data Pointer to the diagnostic data structure.
* @param p_f Pointer to the guiding center particle at the end of the time step.
* @param p_i Pointer to the guiding center particle at the start of the time step.
*/      
void diag_energy_exchange_update_gc(diag_energy_exchange_data* data, 
                                     particle_simd_gc* p_f, 
                                     particle_simd_gc* p_i){
    // Loop over all particles
    GPU_PARALLEL_LOOP_ALL_LEVELS
    for(int i = 0; i < NSIMD; i++) {
        int index = p_f->index[i]; // Get the global index of the particle
        if(p_i->running[i] == 0) continue; // Skip if the particle is not running

        // Common values for the particle.
        real dt = p_f->time[i] - p_i->time[i];
        real w0 = p_i->weight[i];
        real w1 = p_f->weight[i];

        // Computing the Alfven velocity at the particle position.
        real iondens_f, iondens_i;
        plasma_eval_dens(&iondens_f, p_f->rho[i], 
            p_f->r[i], p_f->phi[i], p_f->z[i], p_f->time[i], 1, data->plasma_data);
        plasma_eval_dens(&iondens_i, p_i->rho[i], 
            p_i->r[i], p_i->phi[i], p_i->z[i], p_i->time[i], 1, data->plasma_data);

        // Evaluating the magnetic field strenght at the particle position.
        real B[3];
        real Babs2_f, Babs2_i;
        B_field_eval_B(B, p_f->r[i], p_f->phi[i], p_f->z[i], p_f->time[i], data->B_data);
        Babs2_f = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
        B_field_eval_B(B, p_i->r[i], p_i->phi[i], p_i->z[i], p_i->time[i], data->B_data);
        Babs2_i = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];

        // Evaluating the Alfven velocity at the particle position.
        // Technically we are evaluating the square of the Alfven velocity time mu_0.
        real vA2_f = Babs2_f / (iondens_f * data->thrmass);
        real vA2_i = Babs2_i / (iondens_i * data->thrmass);
        real Babs_f = sqrt(Babs2_f);
        real Babs_i = sqrt(Babs2_i);

        // Relativistic parallel velocity at the particle position.
        real gamma_f = physlib_gamma_ppar(
            p_f->mass[i], p_f->mu[i], p_f->ppar[i], Babs_f);
        real gamma_i = physlib_gamma_ppar(
            p_i->mass[i], p_i->mu[i], p_i->ppar[i], Babs_i);
        real vpar_f = p_f->ppar[i] / (gamma_f * p_f->mass[i]);
        real vpar_i = p_i->ppar[i] / (gamma_i * p_i->mass[i]);

        for (int j = 0; j < data->n_modes; j++) {
            if( data->is_mode_evol[j] == 0 ) continue; // Skip if no evolution for this mode
            real alpha0_f, phipot0_f, theta_f, zeta_f;
            real alpha0_i, phipot0_i, theta_i, zeta_i;
            real sinmhd_f, cosmhd_f, sinmhd_i, cosmhd_i;

            diag_energy_exchange_eval_unscaled_potentials(
                &alpha0_f, &phipot0_f, &theta_f, &zeta_f,
                p_f->r[i], p_f->phi[i], p_f->z[i], j,
                data->boozerdata, data->mhd_data, data->B_data);
            diag_energy_exchange_eval_unscaled_potentials(
                &alpha0_i, &phipot0_i, &theta_i, &zeta_i,
                p_i->r[i], p_i->phi[i], p_i->z[i], j,
                data->boozerdata, data->mhd_data, data->B_data);
            diag_energy_exchange_eval_mode_phase(
                &sinmhd_f, &cosmhd_f, theta_f, zeta_f, p_f->time[i], j,
                data->mhd_data);
            diag_energy_exchange_eval_mode_phase(
                &sinmhd_i, &cosmhd_i, theta_i, zeta_i, p_i->time[i], j,
                data->mhd_data);

            real omega = data->mhd_data->omega_nm[j]; // Toroidal frequency
            real charge = p_i->charge[i]; // Particle charge
            real amplitude = data->mhd_data->amplitude_nm[j];

            /* Unit-amplitude particle-mode coupling. The amplitude is applied
             * only to diagnostics that represent physical energy exchange. */
            real coupling_f = - charge * omega * sinmhd_f
                * (vpar_f * alpha0_f * Babs_f - phipot0_f);
            real coupling_i = - charge * omega * sinmhd_i
                * (vpar_i * alpha0_i * Babs_i - phipot0_i);

            // 1. Evaluating the mode energy exchange.
            real dE_f = amplitude * coupling_f;
            real dE_i = amplitude * coupling_i;
            data->dEnergy[j * data->nprt + index] += 0.5 * (dE_f + dE_i) * dt;

            // 2. Computing the unit-amplitude source term S1.
            data->S1[j * data->nprt + index] += 0.5 * (vA2_f * coupling_f * w1 + \
                                                       vA2_i * coupling_i * w0) * dt;

            // 3. Unit-amplitude quadrature power: the same mode advanced by
            //    a quarter period, i.e. cos(chi) in place of sin(chi).
            real quad_f = - charge * omega * cosmhd_f
                * (vpar_f * alpha0_f * Babs_f - phipot0_f);
            real quad_i = - charge * omega * cosmhd_i
                * (vpar_i * alpha0_i * Babs_i - phipot0_i);
            data->S2[j * data->nprt + index] += 0.5 * (vA2_f * quad_f * w1 + \
                                                       vA2_i * quad_i * w0) * dt;
        }
    }
}

/**
 * @brief Compute the total source terms for the energy exchange diagnostic, 
 * combining all the particle contributions.
 * 
 * This function computes the total source terms for the energy exchange
 * diagnostic by summing the contributions from all particles.
 * 
 * @param data Pointer to the diagnostic data structure.
 * @param S1 Output array for the source term S1.
 * @param S2 Output array for the source term S2.
 * @param clear Whether to clear the source terms after the summation.
 */

void diag_energy_exchange_compact(diag_energy_exchange_data* data, 
                                  real* S1, real* S2, int clear){
    // Check if the diagnostic is enabled
    if (data->enabled == 0) return;

    // Check if the data and source term pointers are valid
    if (data == NULL || S1 == NULL || S2 == NULL) return;

    // Summing all the source terms for all the particles.
    GPU_PARALLEL_LOOP_ALL_LEVELS_REDUCTION(S1, S2)
    for (int i = 0; i < data->nprt; i++) {
        for (int j = 0; j < data->n_modes; j++) {
            real ismodeevol = (real) data->is_mode_evol[j];
            S1[j] += data->S1[j * data->nprt + i] * ismodeevol;
            S2[j] += data->S2[j * data->nprt + i] * ismodeevol;
        }

    }
    
    // If clear is true, we reset the source terms to zero.
    if (clear) {
        memset(data->S1, 0, data->n_modes * data->nprt * sizeof(real));
        memset(data->S2, 0, data->n_modes * data->nprt * sizeof(real));
        memset(data->dEnergy, 0, data->n_modes * data->nprt * sizeof(real));
        fo_cache_reset(data);
    }

}

/** 
 * @brief Update the number of particles in the energy exchange diagnostic.
 * This function updates the number of particles in the diagnostic data structure
 * and reallocates memory for the source terms and energy change arrays.
 * @param data Pointer to the diagnostic data structure.
 * @param nprt New number of particles.
*/
void diag_energy_exchange_update_nprt(diag_energy_exchange_data* data, 
                                        int nprt) {
    // Check if the diagnostic is enabled
    if (data->enabled == 0) return;

    // Removing the data from the GPU.
    GPU_MAP_DELETE_DEVICE(data->S1[0: data->n_modes * data->nprt], 
                          data->S2[0: data->n_modes * data->nprt], 
                          data->dEnergy[0: data->n_modes * data->nprt]);
    if(data->cached_fields_i != NULL) {
        GPU_MAP_DELETE_DEVICE(
            data->cached_fields_i[0: data->n_modes * data->nprt * FO_CACHE_STRIDE]);
    }

    // Free the previous source terms and reallocate for the new number of particles
    free(data->S1);
    free(data->S2);
    free(data->dEnergy);
    free(data->cached_fields_i);

    data->nprt = nprt;
    data->S1 = (real*)malloc(data->n_modes * nprt * sizeof(real));
    data->S2 = (real*)malloc(data->n_modes * nprt * sizeof(real));
    data->dEnergy = (real*)malloc(data->n_modes * nprt * sizeof(real));
    fo_cache_allocate(data);

    memset(data->S1, 0, data->n_modes * nprt * sizeof(real));
    memset(data->S2, 0, data->n_modes * nprt * sizeof(real));
    memset(data->dEnergy, 0, data->n_modes * nprt * sizeof(real));

    // Offloading the data to the GPU again.
    GPU_MAP_TO_DEVICE(data->S1[0: data->n_modes * data->nprt], 
                      data->S2[0: data->n_modes * data->nprt], 
                      data->dEnergy[0: data->n_modes * data->nprt]);
    if(data->cached_fields_i != NULL) {
        GPU_MAP_TO_DEVICE(
            data->cached_fields_i[0: data->n_modes * data->nprt * FO_CACHE_STRIDE]);
    }
}
