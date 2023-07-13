/** @file */
#ifndef __CCL_CONFIG_H_INCLUDED__
#define __CCL_CONFIG_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * Transfer function typedef.
 * Contains all information that describes a specific
 * transfer function. This includes whether there is an
 * emulator being used,
 * if there is a fitting function (E&H is the only option
 * right now), whether to use the BBKS transfer function,
 * and what boltzmann code to use.
 */
typedef enum transfer_function_t
{
  ccl_transfer_none     = 0,
  ccl_eisenstein_hu     = 1,
  ccl_bbks              = 2,
  ccl_boltzmann_class   = 3,
  ccl_boltzmann_camb    = 4,
  ccl_boltzmann_isitgr  = 5,
  ccl_pklin_from_input  = 6,
  ccl_eisenstein_hu_nowiggles = 7,
  ccl_emulator_linpk    = 8,
} transfer_function_t;

/**
 * Matter power spectrum typedef.
 * Contains all information that describes a specific
 * matter power spectrum. This inclues whether we
 * want the linear power spectrum, whether we use
 * halofit, and what halo model is being used.
 */
typedef enum matter_power_spectrum_t
{
    ccl_pknl_none          = 0,
    ccl_linear             = 1,
    ccl_halofit            = 2,
    ccl_halo_model         = 3,
    ccl_pknl_from_input    = 5,
    ccl_pknl_from_boltzman = 6,
    ccl_emulator_nlpk      = 7,
} matter_power_spectrum_t;

/**
 * Configuration typedef.
 * This contains the transfer function,
 * matter power spectrum, and mass function
 * that is being used currently.
 */
typedef struct ccl_configuration {
  transfer_function_t      transfer_function_method;
  matter_power_spectrum_t  matter_power_spectrum_method;
} ccl_configuration;

/**
 * The default configuration object
 * In the default configuration, defined in ccl_core.c
 * CCL runs with:
 * default_config = {ccl_boltzmann_class, ccl_halofit, ccl_nobaryons, ccl_tinker10, ccl_duffy2008}
 */
extern const ccl_configuration default_config;

CCL_END_DECLS

#endif
