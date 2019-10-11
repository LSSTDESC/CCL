from . import ccllib as lib
from .background import growth_factor

def concentration_duffy08_generic(cosmo, M, a, A, B, C):
    M_pivot_inv = cosmo.cosmo.params.h * 5E-13
    return A * (M * M_pivot_inv)**B * a**(-C)

def concentration_duffy08_200mat(cosmo, M, a):
    return concentration_duffy08_generic(cosmo, M, a,
                                         10.14,   # A
                                         -0.081,  # B
                                         -1.01)   # C

def concentration_duffy08_200crit(cosmo, M, a):
    return concentration_duffy08_generic(cosmo, M, a,
                                         5.71,    # A
                                         -0.084,  # B
                                         -0.47)   # C

def concentration_bhattacharya11_generic(cosmo, M, a,
                                         A, B, C):
    gz = growth_factor(cosmo, a)
    M_pivot_inv = cosmo.cosmo.params.h * 2E-14
    nu = (1.12 * (M * M_pivot_inv)**0.3 + 0.53)/gz
    return A * nu**B * gz**C

def concentration_bhattacharya11_200mat(cosmo, M, a):
    return concentration_bhattacharya11_generic(cosmo, M, a,
                                                9.0,    # A
                                                -0.29,  # B
                                                1.15)   # C

def concentration_bhattacharya11_200crit(cosmo, M, a):
    return concentration_bhattacharya11_generic(cosmo, M, a,
                                                5.9,    # A
                                                -0.35,  # B
                                                0.54)   # C
