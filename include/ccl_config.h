#pragma once

typedef enum transfer_function_t
{
    // If using an emulator for P_NL
    emulator          = 0,
    none              = 0,

    fitting_function  = 1,
    eisenstein_hu     = 1,

    bbks              = 2,

    boltzmann         = 3,
    boltzmann_class   = 3,

    boltzmann_camb    = 4
} transfer_function_t;

typedef enum matter_power_spectrum_t
{
    plin           = 0,

    halofit          = 1,
    // more?
    halo_model       = 3
    // even more kinds ...

} matter_power_spectrum_t;

typedef enum mass_function_t
{
    tinker      = 1
} mass_function_t;


typedef struct ccl_configuration {
    transfer_function_t      transfer_function_method;
    matter_power_spectrum_t  matter_power_spectrum_method;
    mass_function_t          mass_function_method;
    // TODO: Halo definition
} ccl_configuration;

// The default configuration object
extern const ccl_configuration default_config;
