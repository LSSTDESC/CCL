from . import ccllib as lib

# Error types
error_types = {
    lib.CCL_ERROR_MEMORY:              'CCL_ERROR_MEMORY',
    lib.CCL_ERROR_LINSPACE:            'CCL_ERROR_LINSPACE',
    lib.CCL_ERROR_INCONSISTENT:        'CCL_ERROR_INCONSISTENT',
    lib.CCL_ERROR_SPLINE:              'CCL_ERROR_SPLINE',
    lib.CCL_ERROR_SPLINE_EV:           'CCL_ERROR_SPLINE_EV',
    lib.CCL_ERROR_INTEG:               'CCL_ERROR_INTEG',
    lib.CCL_ERROR_ROOT:                'CCL_ERROR_ROOT',
    lib.CCL_ERROR_CLASS:               'CCL_ERROR_CLASS',
    lib.CCL_ERROR_COMPUTECHI:          'CCL_ERROR_COMPUTECHI',
    lib.CCL_ERROR_MF:                  'CCL_ERROR_MF',
    lib.CCL_ERROR_HMF_INTERP:          'CCL_ERROR_HMF_INTERP',
    lib.CCL_ERROR_PARAMETERS:          'CCL_ERROR_PARAMETERS',
    lib.CCL_ERROR_NU_INT:              'CCL_ERROR_NU_INT',
    lib.CCL_ERROR_EMULATOR_BOUND:      'CCL_ERROR_EMULATOR_BOUND',
    lib.CCL_ERROR_MISSING_CONFIG_FILE: 'CCL_ERROR_MISSING_CONFIG_FILE',
}
