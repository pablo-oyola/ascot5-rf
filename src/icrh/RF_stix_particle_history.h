#ifndef RF_STIX_PARTICLE_HISTORY_H
#define RF_STIX_PARTICLE_HISTORY_H
#include "../offload.h"
#include "../ascot5.h"
#include "../error.h"
#include "../spline/interp.h" 
#include "../B_field.h"
#include "../boozer.h"
#include "../particle.h"


#define RF_N_HISTORY 3 // Number of history points for the RF particle history
#define RF_N_FIT 128   // Ring-buffer length for the windowed nu(t) fit

typedef struct RF_particle_history{
    real dt[RF_N_HISTORY];    /** Particle history: time step*/
    real bnorm[RF_N_HISTORY]; /** Particle history: magnetic field strength */
    real bphi[RF_N_HISTORY];  /** Particle history: toroidal magnetic field */
    real rhopara[RF_N_HISTORY];  /** Particle history: parallel momentum */
    real R[RF_N_HISTORY];  /** Particle history: radial coordinate */
    real fit_t[RF_N_FIT];    /** Windowed fit: accumulated time of each stored point */
    real fit_vpar[RF_N_FIT]; /** Windowed fit: v_par */
    real fit_kfac[RF_N_FIT]; /** Windowed fit: B_phi / (R |B|), k_par = -n_tor * kfac */
    real fit_Om[RF_N_FIT];   /** Windowed fit: q |B| / m */
    int fit_n;               /** Windowed fit: number of valid points */
    int fit_head;            /** Windowed fit: ring index of the newest point */
    real fit_tacc;           /** Accumulated time since the history was initialised */
    real* last_kick_t;       /** Refractory: time of the last kick per (wave, harmonic) */
    real* last_kick_dur;     /** Refractory: sqrt(t_inter) of the last kick per (wave, harmonic) */
    real *resn;  /** Resonance locations for the Stix diffusion operator ( - kpara * vpara + omega / ntor)*/
    real *resp;  /** Resonance locations for the Stix diffusion operator ( - kpara * vpara + omega / ntor) */
    int nwaves; /** Number of waves in the Stix diffusion operator */
    int lhigh;  /** Maximum number of resonances to store. */
    real* omega; /** Frequencies of the waves */
    int* ntor;  /** Toroidal mode numbers of the waves */
    real qm;  /** Particle history: charge-over-mass ratio */
    real mass; /** Particle mass is required to transform the rhopara into vpara */
} RF_particle_history;

GPU_DECLARE_TARGET_SIMD_UNIFORM(p, nwaves, omega, ntor, lhigh)
void RF_particle_history_init(RF_particle_history* hist, particle_simd_gc* p, 
                              int nwaves, real h,
                              int imrk, real* omega, int* ntor, int lhigh);
GPU_DECLARE_TARGET_SIMD_UNIFORM_END

GPU_DECLARE_TARGET_SIMD_UNIFORM(p, iwave)
void RF_particle_eval_nkicks(RF_particle_history* hist, particle_simd_gc* p, 
                             int imrk, int iwave, int *nkicks,
                             int *lres);
GPU_DECLARE_TARGET_SIMD_UNIFORM_END

GPU_DECLARE_TARGET_SIMD
void RF_particle_history_free(RF_particle_history* hist);
GPU_DECLARE_TARGET_SIMD_END

GPU_DECLARE_TARGET_SIMD_UNIFORM(p)
void RF_particle_history_update(RF_particle_history* hist, particle_simd_gc* p, 
                                int imrk, real h);
GPU_DECLARE_TARGET_SIMD_UNIFORM_END

#endif // RF_STIX_PARTICLE_HISTORY_H