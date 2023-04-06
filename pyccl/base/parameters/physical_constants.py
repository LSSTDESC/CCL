from ... import ccllib as lib
from .parameters_base import Parameters


class PhysicalConstants(Parameters, instance=lib.cvar.constants, freeze=True):
    """Instances of this class hold the physical constants."""
    PI = 3.14_15_92_65_35_89_79_32

    # ~~ ASTRONOMICAL CONSTANTS ~~ #
    # Astronomical Unit, unit conversion (m/au). [exact]
    AU = 149_597_870_800
    # Mean solar day (s/day). [exact]
    DAY = 86400.
    # Sidereal year (days/yr). [IERS2014 in J2000.0]
    YEAR = 365.256_363_004 * DAY

    # ~~ FUNDAMENTAL PHYSICAL CONSTANTS ~~ #
    # Speed of light (m/s). [exact]
    CLIGHT = 299_792_458.
    # Unit conversion (J/eV). [exact]
    EV_IN_J = 1.602_176_634e-19
    ELECTRON_CHARGE = EV_IN_J
    # Electron mass (kg). [CODATA2018]
    ELECTRON_MASS = 9.109_383_701_5e-31
    # Planck's constant (J s). [exact]
    HPLANCK = 6.626_070_15e-34
    # Boltzmann's constant (J/K). [exact]
    KBOLTZ = 1.380_649e-23
    # Universal gravitational constant (m^3/kg/s^2). [CODATA2018]
    GNEWT = 6.674_30e-11

    # ~~ DERIVED CONSTANTS ~~ #
    # Reduced Planck's constant (J s).
    HBAR = HPLANCK / 2 / PI
    # Speed of light (Mpc/h).
    CLIGHT_HMPC = CLIGHT / 1e5
    # Unit conversion (m/pc).
    PC_TO_METER = 180*60*60/PI * AU
    # Unit conversion (m/Mpc).
    MPC_TO_METER = 1e6 * PC_TO_METER
    # Stefan-Boltzmann's constant (kg m^2 / s).
    STBOLTZ = (PI**2/60) * KBOLTZ**4 / HBAR**3 / CLIGHT**2
    # Solar mass in (kg).
    SOLAR_MASS = 4 * PI*PI * AU**3 / GNEWT / YEAR**2
    # Critical density (100 M_sun/h / (Mpc/h)^3).
    RHO_CRITICAL = 3*1e4/(8*PI*GNEWT) * 1e6 * MPC_TO_METER / SOLAR_MASS
    # Neutrino constant required in Omeganuh2.
    NU_CONST = (8*PI**5 * (KBOLTZ/HPLANCK)**3 * (KBOLTZ/15/CLIGHT**3)
                * (8*PI*GNEWT/3) * (MPC_TO_METER**2/CLIGHT**2/1e10))
    # Linear density contrast of spherical collapse.
    DELTA_C = (3/20) * (12*PI)**(2/3)

    # ~~ OTHER CONSTANTS ~~ #
    # Neutrino mass splitting differences.
    # Lesgourgues & Pastor (2012)
    # Adv. High Energy Phys. 2012 (2012) 608515
    # arXiv:1212.6154, p.13
    DELTAM12_sq = 7.62e-5
    DELTAM13_sq_pos = 2.55e-3
    DELTAM13_sq_neg = -2.43e-3
