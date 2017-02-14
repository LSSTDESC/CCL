#pragma once

typedef enum transfer_function_t
{
    // If using an emulator for P_NL
    ccl_emulator          = 0,
    ccl_none              = 0,

    ccl_fitting_function  = 1,
    ccl_eisenstein_hu     = 1,

    ccl_bbks              = 2,

    ccl_boltzmann         = 3,
    ccl_boltzmann_class   = 3,

    ccl_boltzmann_camb    = 4
} transfer_function_t;

typedef enum matter_power_spectrum_t
{
    ccl_linear           = 0,

    ccl_halofit          = 1,
    // more?
    ccl_halo_model       = 3
    // even more kinds ...

} matter_power_spectrum_t;

typedef enum mass_function_t
{
    ccl_tinker      = 1,
    ccl_tinker10    = 2,
    ccl_watson      = 3,
    ccl_angulo      = 4
} mass_function_t;


typedef struct ccl_configuration {
    transfer_function_t      transfer_function_method;
    matter_power_spectrum_t  matter_power_spectrum_method;
    mass_function_t          mass_function_method;
    // TODO: Halo definition
} ccl_configuration;

// The default configuration object
extern const ccl_configuration default_config;
