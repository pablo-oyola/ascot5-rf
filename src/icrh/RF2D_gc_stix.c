/**
 * @file RF2D_gc_stix.c
 * @brief Implementation of the Stix diffusion operator for ICRH
 *
 * stix_data file contains the implementation of the Stix diffusion operator
 * for the ASCOT code, specifically for the 2D representation of the
 * radio-frequency (RF) heating in the context of gyrokinetic simulations.
 * It includes functions for initializing the Stix data, computing
 * resonances, scattering particles, and evaluating fields.
 */

#include <hdf5.h>
#include <hdf5_hl.h>
#include <string.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_errno.h>
#include <math.h>
#include "RF2D_gc_stix.h"
#include "../error.h"
#include "../print.h"
#include "../hdf5io/hdf5_helpers.h"
#include "../ascot5.h"
#include "../B_field.h"
#include "../math.h"
#include "../spline/interp.h"
#include "../particle.h"

#define M_PI 3.14159265358979323846

/**
 * @brief Initializes the RF2D_gc_stix structure with data from an HDF5 file.
 *
 * stix_data function initializes the RF2D_gc_stix structure with the necessary
 * data for the Stix diffusion operator, including wave fields, magnetic field,
 * and other parameters.
 *
 * @param stix_data Pointer to the RF2D_gc_stix structure to be initialized.
 * @param f HDF5 file identifier containing the required data.
 * @param qid Active ID of the RF field to activate.
 * @param lhigh Flag indicating whether high-frequency data is used.
 * @param bdata Pointer to the B_field_data structure containing magnetic field data.
 * @return Error code indicating success or failure of the initialization.
 */
a5err RF2D_gc_stix_init_from_file(RF2D_gc_stix* stix_data, hid_t f, char* qid,
                        int lhigh, B_field_data* bdata){
    // Initialize the Stix data structure with data from the HDF5 file.
    a5err err = 0;
    
    // Check if the stix_data pointer is valid.
    if (stix_data == NULL) {
        print_err("RF2D_gc_stix_init: stix_data pointer is NULL.\n");
        return 1;
    }

    if(bdata == NULL) {
        print_err("RF2D_gc_stix_init: bdata pointer is NULL.\n");
        return 1;
    }

    if(lhigh < 1){
        print_err("RF2D_gc_stix_init: lhigh must be at least 1.\n");
        return 1;
    }

    if (f < 0) {
        print_err("RF2D_gc_stix_init: Invalid file_id.\n");
        return 1;
    }

    #ifdef RFPATHSTIX
    #undef RFPATHSTIX
    #endif
    #define RFPATHSTIX "/RF/RF2D_Stix_XXXXXXXXXX/" 

    // Internal parameters to store temporarily what 
    // we read from the HDF5 file
    char tmp[256];
    int nr, nz;

    /* Read data */
    if( hdf5_read_int(RFPATHSTIX "nr", &nr,
                      f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_int(RFPATHSTIX "nz", &nz,
                      f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_int(RFPATHSTIX "nwaves", &stix_data->nwaves,
                      f, qid, __FILE__, __LINE__) ) {return 1;}

    if(stix_data->nwaves <= 0) {
        print_err("RF2D_gc_stix_init: Number of waves must be positive.\n");
        return 1;
    }

    stix_data->omega = (real*) malloc(stix_data->nwaves * sizeof(real));
    stix_data->ntor = (int*) malloc(stix_data->nwaves * sizeof(real));


    if( hdf5_read_double(RFPATHSTIX "omega", stix_data->omega,
                            f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_int(RFPATHSTIX "ntor", stix_data->ntor,
                        f, qid, __FILE__, __LINE__) ) {return 1;}


    if( hdf5_read_double(RFPATHSTIX "rmin", &stix_data->rmin,
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(RFPATHSTIX "rmax", &stix_data->rmax,
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(RFPATHSTIX "zmin", &stix_data->zmin,
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(RFPATHSTIX "zmax", &stix_data->zmax,
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_int(RFPATHSTIX "include_Eminus", &stix_data->include_Eminus,
                         f, qid, __FILE__, __LINE__) ){
        print_err("RF2D_gc_stix_init: The include_Eminus flag is missing in the HDF5 file: assuming it is 1.\n");
        stix_data->include_Eminus = 1; // Default to 1 if not found
    }
    if( hdf5_read_int(RFPATHSTIX "include_stochastic", &stix_data->include_stochastic,
                         f, qid, __FILE__, __LINE__) ){
        print_err("RF2D_gc_stix_init: The include_stochastic flag is missing in the HDF5 file: assuming it is 1.\n");
        stix_data->include_stochastic = 1; // Default to 1 if not found
    }

    if( hdf5_read_int(RFPATHSTIX "include_vpara_kick", &stix_data->include_vpara_kick,
                         f, qid, __FILE__, __LINE__) ){
        print_err("RF2D_gc_stix_init: The include_vpara_kick flag is missing in the HDF5 file: assuming it is 1.\n");
        stix_data->include_vpara_kick = 1; // Default to 1 if not found
    }
    if( hdf5_read_int(RFPATHSTIX "include_phase_factor", &stix_data->include_phase_factor,
                         f, qid, __FILE__, __LINE__) ){
        print_err("RF2D_gc_stix_init: The include_phase_factor flag is missing in the HDF5 file: assuming it is 1.\n");
        stix_data->include_phase_factor = 1; // Default to 1 if not found
    }
    
    int nsize = nr * nz * stix_data->nwaves;
    real* fields[7]; // Pointers to the fields in the buffer
    char *fieldnames[] = {
        "Eplus_re", "Eplus_im", "Eminus_re", "Eminus_im",
        "kperp", "costheta",  "sintheta"
    };

    for(int i = 0; i < 7; i++) fields[i] = NULL;

    for(int i = 0; i < 7; i++) {
        fields[i] = (real*) malloc(nsize * sizeof(real));
        snprintf(tmp, 256, "%s%s", RFPATHSTIX, fieldnames[i]);
        if( hdf5_read_double(tmp, fields[i],
                             f, qid, __FILE__, __LINE__) ){
            err = 1;
            break;
            }
    }
    if(err) {
        print_err("RF2D_gc_stix_init: Error reading fields from HDF5 file.\n");
        for(int i = 0; i < 7; i++) {
            if(fields[i] != NULL) free(fields[i]);
        }
        return 1;
    }

    // We make some aliases to the fields for easier access.
    real* Eplus_re = fields[0];
    real* Eplus_im = fields[1];
    real* Eminus_re = fields[2];
    real* Eminus_im = fields[3];
    real* kperp = fields[4];
    real* costheta = fields[5];
    real* sintheta = fields[6];

    // Initialize the Stix data structure.
    err = RF2D_gc_stix_init(stix_data, Eplus_re, Eplus_im,
                            Eminus_re, Eminus_im,
                            kperp, costheta, sintheta,
                            stix_data->rmin, stix_data->rmax,
                            stix_data->zmin, stix_data->zmax,
                            nr, nz, stix_data->nwaves,
                            stix_data->include_Eminus,
                            stix_data->include_stochastic,
                            stix_data->include_vpara_kick,
                            stix_data->include_phase_factor,
                            bdata);
    for(int i = 0; i < 7; i++) {
        if(fields[i] != NULL) free(fields[i]);
    }
    if(err) {
        print_err("RF2D_gc_stix_init: Error initializing Stix data from fields.\n");
        return err;
    }

    // Set the B_field_data pointer.
    stix_data->bdata = bdata;

    // Operator numerics (optional; defaults set in RF2D_gc_stix_init).
    real tmpval;
    if( !hdf5_read_double(RFPATHSTIX "fit_window", &tmpval, f, qid, __FILE__, __LINE__) ) {
        stix_data->fit_window = tmpval;
    }
    if( !hdf5_read_double(RFPATHSTIX "kick_refractory", &tmpval, f, qid, __FILE__, __LINE__) ) {
        stix_data->kick_refractory = tmpval;
    }
    print_out(VERBOSE_IO, "RF2D_gc_stix: fit_window = %g s, kick_refractory = %g\n",
              stix_data->fit_window, stix_data->kick_refractory);
    err = 0;
    return err;
}


/**
 * @brief Initializes the RF2D_gc_stix structure with data from provided arrays.
 *
 * RF2D_gc_stix_init_from_data function initializes the RF2D_gc_stix structure
 * with the necessary data for the Stix diffusion operator, including wave fields,
 * magnetic field, and other parameters. It computes the required fields and sets up
 * the interpolation objects for the wave fields.
 *
 * @param stix_data Pointer to the RF2D_gc_stix structure to be initialized.
 * @param Eplus_re Pointer to the real part of the Eplus field.
 * @param Eplus_im Pointer to the imaginary part of the Eplus field.
 * @param Eminus_re Pointer to the real part of the Eminus field.
 * @param Eminus_im Pointer to the imaginary part of the Eminus field.
 * @param kperp Pointer to the kperp field.
 * @param costerm Pointer to the cosine term for the phase factor.
 * @param sinterm Pointer to the sine term for the phase factor.
 * @param rmin Minimum radial coordinate.
 * @param rmax Maximum radial coordinate.
 * @param zmin Minimum vertical coordinate.
 * @param zmax Maximum vertical coordinate.
 * @param nr Number of radial grid points.
 * @param nz Number of vertical grid points.
 * @param nwaves Number of waves in the Stix diffusion operator.
 * @param include_Eminus Flag to include Eminus in the Stix diffusion operator.
 * @param include_stochastic Flag to include stochastic diffusion in the Stix operator.
 * @param include_vpara_kick Flag to include vpara kick in the Stix operator
 * @param include_phase_factor Flag to include phase factor in the Stix operator.
 * @param bdata Pointer to the B_field_data structure containing magnetic field data.
 * @return Error code indicating success or failure of the initialization.
 */
a5err RF2D_gc_stix_init(RF2D_gc_stix* stix_data,
                        real* Eplus_re, real* Eplus_im,
                        real* Eminus_re, real* Eminus_im,
                        real* kperp, real* costerm, real* sinterm,
                        real rmin, real rmax, real zmin, real zmax,
                        int nr, int nz, int nwaves,
                        int include_Eminus, int include_stochastic,
                        int include_vpara_kick, int include_phase_factor,
                        B_field_data* bdata){
    a5err err = 0;
    
    // Allocating memory for the Stix data structure.
    if (stix_data == NULL) {
        print_err("RF2D_gc_stix_init_from_data: stix_data pointer is NULL.\n");
        err = 1;
        return err;
    }
    if(!Eplus_re || !Eplus_im || !Eminus_re || !Eminus_im ||
       !kperp || !costerm || !sinterm) {
        print_err("RF2D_gc_stix_init_from_data: One or more field pointers are NULL.\n");
        err = 1;
        return err;
    }

    if(nr <= 0 || nz <= 0 || nwaves <= 0) {
        print_err("RF2D_gc_stix_init_from_data: nr, nz, and nwaves must be positive.\n");
        return err;
    }

    int nsize = nr * nz * nwaves;
    stix_data->rmin = rmin;
    stix_data->rmax = rmax;
    stix_data->zmin = zmin;
    stix_data->zmax = zmax;
    stix_data->nr = nr;
    stix_data->nz = nz;
    stix_data->nwaves = nwaves;
    stix_data->fit_window = 3.0e-7;   // defaults; RF2D_gc_stix_init_from_file may override
    stix_data->kick_refractory = 1.0;
    stix_data->include_Eminus = include_Eminus;
    stix_data->include_stochastic = include_stochastic;
    stix_data->include_vpara_kick = include_vpara_kick;
    stix_data->include_phase_factor = include_phase_factor;
    stix_data->bdata = bdata;

    real* Eplus2 = (real*) malloc(nsize * sizeof(real));
    real* Eminus2 = (real*) malloc(nsize * sizeof(real));
    real* Ecross2 = (real*) malloc(nsize * sizeof(real));

    if(!Eplus2 || !Eminus2 || !Ecross2) {
        print_err("RF2D_gc_stix_init_from_data: Memory allocation failed.\n");
        return err;
    }

    // Copy the fields into the Stix data structure.
    // Compute the Eplus2, Eminus2 and the cross term.
    #pragma omp parallel for
    for(int i = 0; i < nsize; i++) {
        // 2.0d0 * ( Eplus_re[i]**2 + Eplus_im[i]**2 )
        Eplus2[i] = 2.0 * (Eplus_re[i] * Eplus_re[i] + Eplus_im[i] * Eplus_im[i]);
        // 2.0d0 * ( Eminus_re[i]**2 + Eminus_im[i]**2 )
        Eminus2[i] = 2.0 * (Eminus_re[i] * Eminus_re[i] + Eminus_im[i] * Eminus_im[i]);

        real cos2term = 1.0;
        real sin2term = 0.0;
        if(include_phase_factor) {
            // Re(Eplus2 * conj(Eminus))
            sin2term = 2.0 * costerm[i] * sinterm[i];
            cos2term = 2.0 * costerm[i] * costerm[i] - 1.0;
        }
        Ecross2[i] = + 4.0 * cos2term * ( Eplus_re[i] * Eminus_re[i]   + \
                                          Eplus_im[i] * Eminus_im[i] )   \
                     - 4.0 * sin2term * ( Eplus_re[i] * Eminus_im[i]  - \
                                          Eplus_im[i] * Eminus_re[i] );

    }
    // Set the enabled flag to 1.
    stix_data->enabled = 1;

    // Generating the interpolation objects for the fields.
    err = 0;
    stix_data->Eplus_2 = (interp2D_data*) malloc(sizeof(interp2D_data));
    stix_data->Eminus_2 = (interp2D_data*) malloc(sizeof(interp2D_data));
    stix_data->E2cross = (interp2D_data*) malloc(sizeof(interp2D_data));
    stix_data->kperp = (interp2D_data*) malloc(sizeof(interp2D_data));
    err += interp2Dcomp_setup(stix_data->Eplus_2, Eplus2, nr, nz, 
                              NATURALBC, NATURALBC,
                              rmin, rmax, zmin, zmax);
    err += interp2Dcomp_setup(stix_data->Eminus_2, Eminus2, nr, nz, 
                              NATURALBC, NATURALBC,
                              rmin, rmax, zmin, zmax);
    err += interp2Dcomp_setup(stix_data->E2cross, Ecross2, nr, nz, 
                              NATURALBC, NATURALBC,
                              rmin, rmax, zmin, zmax);
    err += interp2Dcomp_setup(stix_data->kperp, kperp, nr, nz, 
                              NATURALBC, NATURALBC,
                              rmin, rmax, zmin, zmax);

    free(Eplus2);
    free(Eminus2);
    free(Ecross2);

    print_out(VERBOSE_IO, "RF2D_gc_stix_init: Stix data initialized with %d waves.\n", nwaves);
    if(err) {
        print_err("RF2D_gc_stix_init: Error initializing interpolation objects.\n");
        return err;
    }
    print_out(VERBOSE_IO, "RF2D_gc_stix_init: Stix data initialized successfully.\n");
    print_out(VERBOSE_IO, "  Radial grid: %d points from %f to %f\n", nr, rmin, rmax);
    print_out(VERBOSE_IO, "  Vertical grid: %d points from %f to %f\n", nz, zmin, zmax);
    print_out(VERBOSE_IO, "  Number of waves: %d\n", nwaves);
    print_out(VERBOSE_IO, "  Include Eminus: %d\n", include_Eminus);
    print_out(VERBOSE_IO, "  Include stochastic diffusion: %d\n", include_stochastic);
    print_out(VERBOSE_IO, "  Include vpara kick: %d\n", include_vpara_kick);
    print_out(VERBOSE_IO, "  Include phase factor: %d\n", include_phase_factor);

    // Disabling the error raising from the GSL library.
    gsl_set_error_handler_off();
    
    err = 0;
    return err;
}

/**
 * @brief Frees the memory allocated for the RF2D_gc_stix structure.
 *
 * @param stix_data Pointer to the RF2D_gc_stix structure to be freed.
 */
void RF2D_gc_stix_free(RF2D_gc_stix* stix_data){
    // Free the memory allocated for the Stix data structure.
    if (stix_data == NULL) {
        print_err("RF2D_gc_stix_free: stix_data pointer is NULL.");
        return;
    }

    if (stix_data->Eplus_2 != NULL) {
        free(stix_data->Eplus_2->c);
        free(stix_data->Eplus_2);
        stix_data->Eplus_2 = NULL;
    }
    if (stix_data->Eminus_2 != NULL) {
        free(stix_data->Eminus_2->c);
        free(stix_data->Eminus_2);
        stix_data->Eminus_2 = NULL;
    }
    if (stix_data->E2cross != NULL) {
        free(stix_data->E2cross->c);
        free(stix_data->E2cross);
        stix_data->E2cross = NULL;
    }
    if (stix_data->kperp != NULL) {
        free(stix_data->kperp->c);
        free(stix_data->kperp);
        stix_data->kperp = NULL;
    }

    if(stix_data->omega !=NULL) free(stix_data->omega);
    if(stix_data->ntor !=NULL) free(stix_data->ntor);

    // Free resonance arrays.
    if(stix_data->R_resonances != NULL) {
        for(int i = 0; i < stix_data->nwaves; i++) {
            if(stix_data->R_resonances[i] != NULL) {
                free(stix_data->R_resonances[i]);
            }
        }
        free(stix_data->R_resonances);
    }

    if(stix_data->nres != NULL) free(stix_data->nres);
    if(stix_data->res_nums != NULL) {
        for (int i = 0; i < stix_data->nwaves; i++) {
            if(stix_data->res_nums[i] != NULL) {
                free(stix_data->res_nums[i]);
            }
        }
        free(stix_data->res_nums);
    }
    stix_data->res_nums = NULL;
    stix_data->nres = NULL;


    // Free the Stix data structure itself.
    stix_data->enabled = 0;

    // Setting all elements pointer to NULL
    stix_data->Eplus_2 = NULL;
    stix_data->Eminus_2 = NULL;
    stix_data->E2cross = NULL;
    stix_data->kperp = NULL;
    stix_data->omega = NULL;
    stix_data->ntor = NULL;
}


/**
 * @brief Offloads the RF2D_gc_stix structure to the GPU or other device.
 *
 * @param stix_data Pointer to the RF2D_gc_stix structure to be offloaded.
 */
void RF2D_gc_stix_offload(RF2D_gc_stix* stix_data){
    // Offload the Stix data structure to the GPU or other device.
    if (stix_data == NULL) {
        print_err("RF2D_gc_stix_offload: stix_data pointer is NULL.");
        return;
    }

    int nsize = stix_data->nr * stix_data->nz * stix_data->nwaves * NSIZE_COMP2D;
    GPU_MAP_TO_DEVICE(stix_data[0:1],
                      stix_data->Eplus_2, stix_data->Eminus_2,
                      stix_data->E2cross, stix_data->kperp,
                      stix_data->omega[0:stix_data->nwaves],
                      stix_data->ntor[0:stix_data->nwaves],
                      stix_data->Eplus_2->c[0:nsize],
                      stix_data->Eminus_2->c[0:nsize],
                      stix_data->E2cross->c[0:nsize],
                      stix_data->kperp->c[0:nsize],
                      stix_data->R_resonances[0:stix_data->nwaves],
                      stix_data->res_nums[0:stix_data->nwaves],
                      stix_data->nres[0:stix_data->nwaves]
                    );

    for (int i = 0; i < stix_data->nwaves; i++) {
        int nres = stix_data->nres[i];
        if (nres <= 0) continue; // Skip if no resonances
        GPU_MAP_TO_DEVICE(stix_data->R_resonances[i][0:stix_data->nres],
                          stix_data->res_nums[i][0:stix_data->nres]);
    }
}

void RF2D_gc_stix_scale_amplitude(RF2D_gc_stix* stix_data, real factor){
    if(!stix_data) return;
    if(!isfinite(factor)) return;
    interp2D_data* arr[] = { stix_data->Eplus_2, stix_data->Eminus_2, stix_data->E2cross };
    for(int i = 0; i < 3; i++){
        if(!arr[i] || !arr[i]->c) continue;
        if(arr[i]->n_x <= 0 || arr[i]->n_y <= 0) continue;
        int nsize = arr[i]->n_x * arr[i]->n_y * (int)NSIZE_COMP2D;
        if(nsize <= 0) continue;
        for(int j = 0; j < nsize; j++)
            arr[i]->c[j] *= factor;
    }
}

/**
 * @brief Computes the cold resonance locations based on the frequencies and the
 * magnetic field strenght.
 * 
 * It will look for the cold resonance condition, i.e., the condition ignoring
 * the Doppler shift. The resonance condition is given by:
 * \[ \omega_\mathrm{RF} = l_i \Omega_\mathrm{cyclotron} \]
 * being l_i the resonance number, which is an integer.
 * 
 * The purpose of this function is to compute the radial locations of the cold
 * resonances, assuming approximately that B \propto 1/R.
 *
 * @param stix_data Pointer to the RF2D_gc_stix structure to be initialized.
 * @param bdata Pointer to the B_field_data structure containing magnetic field data.
 * @param n_max_res Maximum number of resonance locations to compute.
 * @param qm Charge-to-mass ratio of the particles.
 */
a5err RF2D_gc_stix_compute_cold_resonances(RF2D_gc_stix* stix_data,
                                           B_field_data* bdata, int n_max_res,
                                           real qm){
    if(stix_data == NULL || bdata == NULL) {
        print_err("RF2D_gc_stix_compute_cold_resonances: stix_data or bdata pointer is NULL.");
        return 1;
    }

    if(n_max_res <= 0) {
        print_err("RF2D_gc_stix_compute_cold_resonances: n_max_res must be positive.");
        return 1;
    }

    stix_data->n_max_res = n_max_res;
    stix_data->R_resonances = (real**) malloc(stix_data->nwaves * sizeof(real*));
    stix_data->nres = (int*) malloc(stix_data->nwaves * sizeof(int));
    stix_data->res_nums = (int**) malloc(stix_data->nwaves * sizeof(int*));

    // Checking that we are dealing with a 2D magnetic field, as otherwise the resonances
    // would not be only function of the major radius coordinate.
    if(bdata->type != B_field_type_2DS) {
        print_err("RF2D_gc_stix_compute_cold_resonances: B_field_data must be of type B_field_type_2DS.");
        // Creating dummy data.
        for(int i = 0; i < stix_data->nwaves; i++) {
            stix_data->R_resonances[i] = (real*) malloc(n_max_res * sizeof(real));
            stix_data->res_nums[i] = (int*) malloc(n_max_res * sizeof(int));
            stix_data->nres[i] = n_max_res; // All the waves should look for the same number of resonances.
            for(int j = 0; j < n_max_res; j++) {
                stix_data->R_resonances[i][j] = 0.0; // Dummy value
                stix_data->res_nums[i][j] = j + 1; // Resonance number
            }
        }

        return 0;
    }

    real Rzaxis[2];
    B_field_get_axis_rz(&Rzaxis[0], bdata, 0.0); // Get the axis R and Z coordinates
    real zaxis = Rzaxis[1]; 
    
    real* Rgrid = (real*) malloc(1024 * sizeof(real));
    real Btmp[3];
    real* Bnorm = (real*) malloc(1024 * sizeof(real));
    if(Rgrid == NULL || Bnorm == NULL) {
        print_err("RF2D_gc_stix_compute_cold_resonances: Memory allocation failed for Rgrid or Bnorm.");
        return 1;
    }
    for(int i = 0; i < 1024; i++) {
        Rgrid[i] = stix_data->rmin + i * (stix_data->rmax - stix_data->rmin) / 1023.0;

        // Evaluating the local magnetic field strength at the point.
        B_field_eval_B(&Btmp[0], Rgrid[i], 0.0, zaxis, 0.0, bdata);

        Bnorm[i] = sqrt(Btmp[0] * Btmp[0] + Btmp[1] * Btmp[1] + Btmp[2] * Btmp[2]);
    }

    real* Rres_tmp = (real*) malloc(n_max_res * sizeof(real));
    int*  res_indx = (int*)  malloc(n_max_res * sizeof(int));
    int maxreson = 0;
    memset(Rres_tmp, 0, n_max_res * sizeof(real));
    memset(res_indx, 0, n_max_res * sizeof(int));

    for (int iwave=0; iwave < stix_data->nwaves; iwave++){
        int nres = 0;
        for(int i = 0; i < n_max_res; i++){
            int l_res = i + 1; // Resonance number
            real prev = stix_data->omega[iwave] - l_res * qm * Bnorm[0];
            real next = stix_data->omega[iwave] - l_res * qm * Bnorm[0];

            // We loop through the radial coordinate to find the point where there is a
            // change of sign in the resonance condition -> Resonance found. Then
            // we perform linear interpolation to find the exact resonance location.
            for(int j = 1; j < 1024; j++) {
                prev = next;
                next = stix_data->omega[iwave] - l_res * qm * Bnorm[j];
                if(prev*next < 0.0){
                    // We have a root between j-1 and j: linear interpolation
                    // of the sign-changing segment (prev, next) to zero.
                    real Rres = Rgrid[j-1] + fabs(prev) / (fabs(prev) + fabs(next)) * (Rgrid[j] - Rgrid[j-1]);
                    Rres_tmp[i] = Rres;
                    res_indx[i] = l_res; // Store the index of the root
                    nres++;
                    break; // We found a resonance, break the loop.
                }
            }
        }
        // We store the total resonance number.
        stix_data->nres[iwave] = nres;
        maxreson = fmax(maxreson, nres);

        stix_data->R_resonances[iwave] = (real*) malloc(nres * sizeof(real));
        stix_data->res_nums[iwave] = (int*) malloc(nres * sizeof(int));

        // We fill the resonance locations and numbers.
        int idx_res = 0;
        for(int i = 0; i < n_max_res; i++) {
            if(res_indx[i] > 0) {
                stix_data->R_resonances[iwave][idx_res] = Rres_tmp[i];
                stix_data->res_nums[iwave][idx_res] = res_indx[i];
                idx_res++;
            }
        }

        // Reset the temporary arrays for the next wave.
        memset(Rres_tmp, 0, n_max_res * sizeof(real));
        memset(res_indx, 0, n_max_res * sizeof(int));
    }
    stix_data->n_max_res = 0;

    // Debugging the resonances found.
    print_out(VERBOSE_IO, "RF2D_gc_stix_compute_cold_resonances: Cold resonances found:\n");
    for (int iwave=0; iwave < stix_data->nwaves; iwave++){
        print_out(VERBOSE_IO, " > Wave %d: found %d resonances:\n", iwave, stix_data->nres[iwave]);
        for(int i = 0; i < stix_data->nres[iwave]; i++) {
            print_out(VERBOSE_IO, "   >> Resonance l=%d: R = %f\n", stix_data->res_nums[iwave][i], stix_data->R_resonances[iwave][i]);
            stix_data->n_max_res = fmax(stix_data->n_max_res, stix_data->res_nums[iwave][i]);
        }
    }

    // Free the temporary arrays.
    free(Rgrid);
    free(Bnorm);
    free(Rres_tmp);
    free(res_indx);
    return 0;
}

/**
 * @brief Least-squares parabola through the history points of the last
 *        stix_data->fit_window seconds, giving nudot and nudot2 at the crossing.
 *
 * y(u) = c0 + c1 u + c2 u^2 with u = (t - t_now)/W in [-1, 0] and
 * y = nu - omega, nu = k_par v_par + l Omega. nudot is evaluated at the root
 * of y nearest u = 0 (the crossing), nudot2 = 2 c2 / W^2. The 3-point second
 * difference of the post-collision history is dominated by collisional noise
 * (~dt^(-3/2)) at ~10 ns steps; the window fit's noise is set by W instead.
 *
 * @return 1 when at least three points fell in the window, 0 otherwise.
 */
static int RF2D_gc_stix_fit_derivs(RF2D_gc_stix* stix_data, RF_particle_history* hist,
                                   int iwave, int l, real* nudot_out, real* nudot2_out) {
    const real W = stix_data->fit_window;
    if(W <= 0.0 || hist->fit_n < 3) return 0;
    real t_now = hist->fit_t[hist->fit_head];
    real s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, y0 = 0, y1 = 0, y2 = 0, umin = 0;
    int n = 0;
    for(int i = 0; i < hist->fit_n; i++) {
        int idx = (hist->fit_head - i + RF_N_FIT) % RF_N_FIT;
        real tau = hist->fit_t[idx] - t_now;
        if(tau < -W) break;
        real u = tau / W, u2 = u * u;
        real nu = -stix_data->ntor[iwave] * hist->fit_kfac[idx] * hist->fit_vpar[idx]
                  + l * hist->fit_Om[idx];
        real y = nu - stix_data->omega[iwave];
        s0 += 1.0; s1 += u; s2 += u2; s3 += u2 * u; s4 += u2 * u2;
        y0 += y; y1 += y * u; y2 += y * u2;
        n++;
        if(u < umin) umin = u;
    }
    if(n < 3) return 0;
    real a11 = s0, a12 = s1, a13 = s2, a22 = s2, a23 = s3, a33 = s4;
    real det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13)
               + a13 * (a12 * a23 - a22 * a13);
    if(det == 0.0) return 0;
    real c0 = (y0 * (a22 * a33 - a23 * a23) - a12 * (y1 * a33 - a23 * y2)
               + a13 * (y1 * a23 - a22 * y2)) / det;
    real c1 = (a11 * (y1 * a33 - a23 * y2) - y0 * (a12 * a33 - a23 * a13)
               + a13 * (a12 * y2 - y1 * a13)) / det;
    real c2 = (a11 * (a22 * y2 - y1 * a23) - a12 * (a12 * y2 - y1 * a13)
               + y0 * (a12 * a23 - a22 * a13)) / det;
    real u0 = 0.0;
    if(c2 != 0.0) {
        real disc = c1 * c1 - 4.0 * c2 * c0;
        if(disc >= 0.0) {
            real sq = sqrt(disc);
            real r1 = (-c1 + sq) / (2.0 * c2), r2 = (-c1 - sq) / (2.0 * c2);
            int ok1 = (r1 <= 0.0 && r1 >= umin), ok2 = (r2 <= 0.0 && r2 >= umin);
            if(ok1 && (!ok2 || fabs(r1) < fabs(r2))) u0 = r1;
            else if(ok2) u0 = r2;
        }
    } else if(c1 != 0.0) {
        real r = -c0 / c1;
        if(r <= 0.0 && r >= umin) u0 = r;
    }
    *nudot_out = fmax(1e-14, fabs((c1 + 2.0 * c2 * u0) / W));
    *nudot2_out = fmax(1e-14, fabs(2.0 * c2 / (W * W)));
    return 1;
}

/**
 * @brief Computes the interaction time for a particle with the RF wave during resonance.
 * 
 * This function calculates the squared interaction time for a particle with the RF wave
 * during resonance, based on the particle's history and the Stix diffusion operator.
 * It uses the particle's gyrofrequency and parallel momentum to compute the resonance
 * condition derivatives and the interaction time. 
 * 
 * The interaction time is based on the fixed-point technique to integrate a fast oscillating
 * integral. See more details in "M. Choi, PPCF (2010)".
 * 
 * @param stix_data Pointer to the RF2D_gc_stix structure containing the Stix data.
 * @param hist Pointer to the RF_particle_history structure containing the particle's history.
 * @param iwave Index of the wave for which the interaction time is computed.
 * @return The squared interaction time for the particle with the RF wave, or -1.
 *         If the interaction time cannot be computed, it returns -1.
 */
real RF2D_gc_stix_get_interaction_time(RF2D_gc_stix* stix_data, 
                                       RF_particle_history* hist, 
                                       int iwave, int l){
    real time = -1.0; // Default value indicating no interaction time computed.

    const int curr = 0; // Current index
    const int prev = 1; // Previous index
    const int prev_prev = 2; // Previous-previous index

    // Time step calculations
    real ddt = hist->dt[curr] + hist->dt[prev];
    real diff_dt = hist->dt[curr] - hist->dt[prev];
    // Parallel wave vector at each of the stored points: the toroidal wave
    // vector projected on the local field direction, k_par = -(n_tor/R) B_phi/|B|
    // (minus: TORIC's toroidal angle is varphi = -phi, Brambilla 1999, Eq. (55)).
    // Evaluating it at every point also retains the v_para * dk_par/dt part of
    // the derivatives below, which is dropped if k_par is frozen at the current
    // position.
    real kpara_curr = -stix_data->ntor[iwave] / hist->R[curr] * hist->bphi[curr] / hist->bnorm[curr];
    real kpara_prev = -stix_data->ntor[iwave] / hist->R[prev] * hist->bphi[prev] / hist->bnorm[prev];
    real kpara_prev_prev = -stix_data->ntor[iwave] / hist->R[prev_prev] * hist->bphi[prev_prev] / hist->bnorm[prev_prev];

    real nu_curr = kpara_curr * hist->rhopara[curr] * hist->bnorm[curr] / hist->mass + hist->qm * l * hist->bnorm[curr];
    real nu_prev = kpara_prev * hist->rhopara[prev] * hist->bnorm[prev] / hist->mass + hist->qm * l * hist->bnorm[prev];
    real nu_prev_prev = kpara_prev_prev * hist->rhopara[prev_prev] * hist->bnorm[prev_prev] / hist->mass + hist->qm * l * hist->bnorm[prev_prev];

    // Evaluation of the \dot\nu and \ddot\nu terms at the resonance crossing:
    // 1. The curve of \nu(t) is approximated by a parabola and the coefficients
    //    are computed using the three points available.
    // 2. The crossing via \nu(t_0) = 0 is computed.
    // 3. The derivatives \dot\nu and \ddot\nu are evaluated at t_0.
    // 4. The interaction time is computed.
    real a_coef = ( (nu_prev_prev - nu_prev) * hist->dt[curr]
                   + (nu_curr     - nu_prev) * hist->dt[prev] )
                  / ( hist->dt[prev] * hist->dt[curr] * ddt );
    real b_coef = ( nu_curr - nu_prev_prev ) / ddt - a_coef * diff_dt;
    real nudot2 = 2.0 * a_coef; // \ddot\nu, constant along the parabola

#ifdef RF2D_GC_STIX_LEGACY_NUDOT
    real ddt2 = hist->dt[curr]*hist->dt[curr] + hist->dt[prev]*hist->dt[prev];
    real nudot = ( nu_curr - nu_prev_prev ) / ddt; // pre-review: \dot\nu at t_prev
    nudot2 = 2.0 / ddt2 * (
        nu_curr - 2.0 * nu_prev + nu_prev_prev - nudot * diff_dt
    );
#else
    real nudot = b_coef; // fallback: \dot\nu at t_prev (pre-review value)
    if (fabs(a_coef) > 1e-14) {
        // Solve a*tau^2 + b*tau + c = 0 (c = nu_prev - omega) for the root nearest
        // tau = 0 that also falls inside the sampled bracket
        // [-hist->dt[prev], hist->dt[curr]]; that is the physical crossing
        // t_0, the other root of the parabola is spurious. A negative
        // discriminant means the local parabola does not actually recross
        // zero (e.g. noisy or poorly-resolved history) and we keep the
        // t_prev fallback above rather than propagate a NaN.
        real c_coef = nu_prev - stix_data->omega[iwave]; // the crossing is nu = omega
        real disc = b_coef * b_coef - 4.0 * a_coef * c_coef;
        if (disc >= 0.0) {
            real sq = sqrt(disc);
            real q = -0.5 * ( b_coef + copysign(sq, b_coef) );
            real tau_a = (fabs(q) > 1e-300) ? q / a_coef : 0.0;
            real tau_b = (fabs(q) > 1e-300) ? c_coef / q : tau_a;
            int a_in = (tau_a >= -hist->dt[prev] && tau_a <= hist->dt[curr]);
            int b_in = (tau_b >= -hist->dt[prev] && tau_b <= hist->dt[curr]);
            real tau0 = tau_a;
            if (b_in && (!a_in || fabs(tau_b) < fabs(tau_a))) tau0 = tau_b;
            if (a_in || b_in) nudot = 2.0 * a_coef * tau0 + b_coef;
        }
    }
#endif

    nudot = fmax(1e-14, fabs(nudot)); // Avoiding zero nudot
    nudot2 = fmax(1e-14, fabs(nudot2)); // Avoiding zero nudot2

    // Windowed least-squares derivatives replace the 3-point ones whenever at
    // least three history points fall in stix_data->fit_window (see above).
    {
        real fnd, fnd2;
        if(RF2D_gc_stix_fit_derivs(stix_data, hist, iwave, l, &fnd, &fnd2)) {
            nudot = fnd;
            nudot2 = fnd2;
        }
    }

    // The two expressions below are the two limits of the same oscillating
    // integral: the stationary-phase one, valid when the particle crosses the
    // resonance transversally, and the Airy one, valid when the crossing is
    // tangential and the two stationary points coalesce.
    //
    // Normalisation. With I = int exp(i Phi(t)) dt the phase integral over a
    // crossing, the gyro-averaged drift is dmu = q^2 E_c^2 J^2 |I|^2 / (2 m B),
    // E_c = |Ex + i Ey| / 2 the rotating-field amplitude. The kick in
    // RF2D_gc_stix_scatter is scaling2 * term1 * t_inter / (4 B m) with
    // term1 = 2|E+|^2 J^2 = |Ex + i Ey|^2 J^2 = 4 E_c^2 J^2 (the 1/4 undoes that
    // factor; the 1/2 of mu = m v_perp^2 / 2B is NOT in the prefactor), so the
    // value returned here is t_inter = |I|^2 / 2:
    //   transverse crossing (stationary phase, |I|^2 = 2 pi/|nudot|): pi/|nudot|,
    //   tangential passage (Airy, |I_pair|^2 = tau_c^2 for the coalesced entry/exit
    //   pair): 0.5 tau_c^2, applied once per passage (see kick_refractory in
    //   RF2D_gc_stix_scatter). Its large-|zeta| mean is twice the single-crossing
    //   value, as it must be for two incoherent crossings.
    // The same chain in ORBIT-RF (scatrf: pi/|wdot|, code units) agrees with
    // this to 0.04 %; the earlier 2 pi/|nudot| doubled the drift and, through
    // dmu_stoch, the diffusion (AUG #40412 GC vs FO, Oct 2026).
    //
    // Switch. Written in units of (2/|nudot2|)^(2/3), the two branches are
    // 2 pi^2 Ai(zeta)^2 and pi |zeta|^(-1/2); they are equal where
    // 2 pi Ai(-x)^2 sqrt(x) = 1, at x = 0.497 and x = 1.792. Switching at the
    // second root keeps the coherent first lobe of Ai^2 (max 2.83 at x = 1.02)
    // and hands over to the incoherent value where the phase accumulated
    // between the two crossings, (4/3)|zeta|^(3/2), reaches ~pi, i.e. where an
    // ensemble of markers starts to wash the interference out. The branches
    // are continuous there. The first root is the alternative continuous
    // choice that discards the lobe. (Full-orbit pairs at |zeta| 2-7 scatter
    // between 0.3 and 1.7 x the incoherent sum: the real interference phase
    // is set by int nu dt between the crossings, which no local form gives.) Selecting on |zeta| rather than on
    // |nudot| > 0.5 |nudot2| dt_sim (ORBIT) is what makes the switch
    // independent of the integration step: the ORBIT form only coincides
    // with the crossover when dt_sim ~ 2.5 nudot2^(-1/3), which holds for its
    // hundreds-of-ns steps and not for ASCOT's ~10 ns ones, where it would
    // leave near-tangential crossings on pi/|nudot|, which diverges.
    const real zeta_switch = 1.792033;
    real zeta = - (nudot * nudot) / pow(2.0 * nudot2 * nudot2, 2.0/3.0);

    // Squared interaction time
    if (fabs(zeta) > zeta_switch) {
        time = M_PI / fabs(nudot);
    } else {
        real airy_val = gsl_sf_airy_Ai(zeta, GSL_PREC_DOUBLE);
        time = 2.0 * M_PI * airy_val / pow(fabs(nudot2 / 2.0), 1.0/3.0);
        time = 0.5 * time * time;
    }

    return time;
}

/**
 * @brief Evaluates the RF fields at given coordinates and time.
 * 
 * Uses the interpolation objects to compute the electric fields, taking into
 * account the internal flags.
 * 
 * @param r Radial coordinate.
 * @param phi Toroidal angle coordinate (not used in this function).
 * @param z Vertical coordinate.
 * @param t Time coordinate (not used in this function).
 * @param stix_data Pointer to the RF2D_gc_stix structure containing the Stix data.
 * @param Eplus_2 Variable to store the squared Eplus field.
 * @param Eminus_2 Variable to store the squared Eminus field (if included).
 * @param E2cross Variable to store the cross term of the electric fields.
 * @param kperp Variable to store the perpendicular wave vector.
 * @return a5err Error code indicating success or failure of the evaluation.
 */
a5err RF2D_gc_stix_eval_fields(real r, real phi, real z, real t,
                              RF2D_gc_stix* stix_data,
                              real* Eplus_2, real* Eminus_2,
                              real* E2cross, real* kperp){
    // Setting the values to 0.
    *Eplus_2 = 0.0;
    *Eminus_2 = 0.0;
    *E2cross = 0.0;
    *kperp = 0.0;

    a5err err = 0;

    // Evaluate the fields at the given coordinates and time.
    if (!stix_data->enabled) return err;

    // Evaluating the fields using the interpolation objects.
    err += interp2Dcomp_eval_f(Eplus_2, stix_data->Eplus_2, r, z);
    if(stix_data->include_Eminus) err += interp2Dcomp_eval_f(Eminus_2, stix_data->Eminus_2, r, z);
    err += interp2Dcomp_eval_f(E2cross, stix_data->E2cross, r, z);
    err += interp2Dcomp_eval_f(kperp, stix_data->kperp, r, z);

    return err;
}


/**
 * @brief Applies the Stix scattering operator to a particle and updates its internal state.
 * 
 * @param stix Array with the Stix data for the RF field.
 * @param hist Pointer to the RF_particle_history structure containing the particle's history.
 * @param p Pointer to the particle_simd_gc structure containing the current particle state.
 */
void RF2D_gc_stix_scatter(RF2D_gc_stix* stix, RF_particle_history* hist,
                          particle_simd_gc* p, real* h, real* rnd, uint8* used){
    // Apply the Stix scattering operator to the particle and update its internal state.
    if (stix == NULL || hist == NULL || p == NULL) {
        print_err("RF2D_gc_stix_scatter: One or more input pointers are NULL.\n");
        return;
    }

    if (!stix->enabled) return; // Stix operator is not enabled.

    // Updating the internal status of the particles.
    #pragma omp simd
    for(int imrk=0; imrk < NSIMD; imrk++) {
        if(p->running[imrk] == 0) continue; // Particle is not running, skip.
        int imrk_rnd_idx = 2 * stix->n_max_res * stix->nwaves * imrk;
        RF_particle_history_update(&hist[imrk], p, imrk, h[imrk]);

        // Checking for all the waves whether there's been some resonance.
        for(int iwave=0; iwave < stix->nwaves; iwave++) {
            // Evaluating the number of kicks.
            int nkicks, lres;
            RF_particle_eval_nkicks(&hist[imrk], p, imrk, iwave, &nkicks, &lres);

            if(nkicks <= 0 || lres == -1) continue; // No kicks, go to the next wave.

            // We have a resonance crossing. Let's evaluate the fields.
            real Eplus_2 = 0.0;
            real Eminus_2 = 0.0;
            real E2cross = 0.0;
            real kperp = 0.0;

            // Getting the resonance time.
            real t_inter = RF2D_gc_stix_get_interaction_time(stix, &hist[imrk], iwave, lres);
            if(t_inter < 0.0 || !isfinite(t_inter)) continue; // No interaction time, go to the next wave.

            // One quasilinear kick per crossing: the kick itself (drhopara and the
            // stochastic dmu) can push nu back through omega within the same
            // passage, which fires further kicks; skip sign changes closer than
            // kick_refractory * sqrt(t_inter) to the last kick on this harmonic.
            if(stix->kick_refractory > 0.0) {
                int kidx = iwave * hist[imrk].lhigh + (lres - 1);
                real tnow = hist[imrk].fit_tacc;
                if(tnow - hist[imrk].last_kick_t[kidx]
                   < stix->kick_refractory * hist[imrk].last_kick_dur[kidx]) continue;
                hist[imrk].last_kick_t[kidx] = tnow;
                hist[imrk].last_kick_dur[kidx] = sqrt(t_inter);
            }

            RF2D_gc_stix_eval_fields(p->r[imrk], p->phi[imrk], p->z[imrk], p->time[imrk],
                                     stix, &Eplus_2, &Eminus_2, &E2cross, &kperp);
            if(kperp == 0.0) continue; // Outside domain.

            // Evaluating so prefactors.
            for(int ikick = 0; ikick < nkicks; ikick++){
                real rhoL = 1/ p->charge[imrk] * sqrt(2.0 * p->mu[imrk] * p->mass[imrk] / hist[imrk].bnorm[0]);
                real arg = fabs(kperp * rhoL);

                // Evaluating the Bessel functions
                gsl_sf_result rJm1, rJl, rJp1;
                real Jm1, Jl, Jp1; // Bessel functions of order lres-1, lres and lres+1
                int err1, err2, err3;
                err1 = gsl_sf_bessel_Jn_e(lres - 1, arg, &rJm1);
                err2 = gsl_sf_bessel_Jn_e(lres, arg, &rJl);
                err3 = gsl_sf_bessel_Jn_e(lres + 1, arg, &rJp1);
                if(err1 || err2 || err3){
                    p->err[imrk] = error_raise( ERR_BESSEL_EVALUATION, __LINE__, EF_RF_GC2D );
                    p->running[imrk] = 0; // Stop the particle.
                    continue;
                }
                Jm1 = rJm1.val;
                Jl = rJl.val;
                Jp1 = rJp1.val;

                // We evaluate the value of the derivatives using the recurrence relations.
                real dbessel_l_m1 = 0.5 * (lres - 1) * Jm1 - 0.5*arg*Jl;
                real dbessel_l_p1 = 0.5 * arg * Jl - 0.5*(lres + 1) * Jp1;

                real term1 = Eplus_2 * Jm1 * Jm1 + \
                             Eminus_2 * Jp1 * Jp1 + \
                             E2cross * Jm1 * Jp1;
                real term2 = 2 * Eplus_2 * Jm1 * dbessel_l_m1 + \
                             2 * Eminus_2 * Jp1 * dbessel_l_p1 + \
                             E2cross * (Jm1 * dbessel_l_p1 + Jp1 * dbessel_l_m1);
                
                real omega_cycl = hist[imrk].qm * hist[imrk].bnorm[0];
                real scaling_para = lres * omega_cycl / stix->omega[iwave] * p->charge[imrk];
                real scaling2 = scaling_para * scaling_para;

                // Kick in the magnetic moment.
                real dmu = scaling2 * (term1 + term2) * t_inter / (4 * hist[imrk].bnorm[0] * p->mass[imrk]);
                real kpara = -stix->ntor[iwave] / hist[imrk].R[0]
                             * hist[imrk].bphi[0] / hist[imrk].bnorm[0];
                real drhopara = kpara / (lres * omega_cycl) * dmu;

                // Computing the stochastic contribution. sqrt(2) normalizes
                // <irnd^2> to 1 over the random phase (cos(...) alone gives
                // 1/2), which is required for the realized diffusion
                // 0.5*<dmu_stoch^2> to match d/dmu of the drift dmu above -
                // see tests/rf/test_stix_operator.py::TestDriftDiffusionConsistency.
                real irnd = sqrt(2.0) * cos(lres * rnd[imrk_rnd_idx] * 2.0 * M_PI);
                used[imrk_rnd_idx] = 0; // Mark as used.

                // The following part is prone to underflow since the mu values are always
                // scaled by the mass of the particle. We temporarily remove the charge.
                dmu /= p->charge[imrk];
                real tmp_mu = p->mu[imrk] / p->charge[imrk];
                real dmu_stoch = 0.0;
                if(term1 + term2 != 0.0) {
                    dmu_stoch = sqrt(fabs(2 * tmp_mu * term1 / (term1 + term2) * dmu)) * irnd;
                }
                imrk_rnd_idx++; // For next fetch.
                real drhopara_stoch = kpara / (lres * omega_cycl) * dmu_stoch;

                // We recover the original scaling.
                dmu *= p->charge[imrk];
                dmu_stoch *= p->charge[imrk];
                drhopara_stoch *= p->charge[imrk];

                // Updating the values of the parallel momentum and the magnetic moment.
                p->mu[imrk] +=  dmu_stoch + dmu;
                p->ppar[imrk] += (drhopara + drhopara_stoch) * hist[imrk].bnorm[0];
            }
        }
    }
}

/**
 * @brief From the particle queue, it gets the value of the charge
 * over mass ratio, that it will be used to compute the cold resonances.
 * 
 * This function will check whether all particles in the queue have the same
 * charge-over-mass ratio only for valid particles.
 * 
 * @param pq Particle queue to check.
 * @return real Value of the charge-over-mass ratio, or -1 if there's an error.
 */
real guess_qm(particle_queue* pq){
    if (pq == NULL) {
        print_err("guess_qm: pq pointer is NULL.\n");
        return -1.0;
    }
    real qm = -1.0;
    for(int i = 0; i < pq->n; i++) {
        if(pq->p[i]->mass <= 0.0) continue; // Invalid particle, skip.
        if(pq->p[i]->charge == 0.0) continue; // Neutral particle, skip.
        real qm_i = fabs(pq->p[i]->charge / pq->p[i]->mass);

        if(qm < 0.0) {
            qm = qm_i; // First valid particle, set qm.
            continue;
        }
        if(fabs(qm - qm_i) > 1e-3 * qm){
            print_err("guess_qm: Particles in the queue have different charge-over-mass ratios: %e vs %e.\n", qm, qm_i);
            return -1.0;
        }
        // They are the same, continue.
        qm = qm_i;
        
    }
    return qm;
}


/**
 * @brief Sets the usage flag for random numbers for a given particle.
 * 
 * This function will set the usage flag for all random numbers associated
 * with a specific particle in the Stix data structure.
 * 
 * @param stix_data Structure with the Stix data.
 * @param used Pointer to the array of usage flags.
 * @param imrk Index of the particle in the queue.
 * @param value Value to set the usage flag.
 */
void set_rndusage(RF2D_gc_stix* stix_data, uint8* used, int imrk, uint8 value){
    if(stix_data == NULL || used == NULL) {
        print_err("set_rndusage: stix_data or used pointer is NULL.\n");
        return;
    }
    int nflags = 2 * stix_data->n_max_res * stix_data->nwaves;
    int start_idx = nflags * imrk;
    for(int i = 0; i < nflags; i++) {
        used[start_idx + i] = value;
    }
}