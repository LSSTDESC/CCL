"""
Code: Test for the AMEMU Linear Matter Power Spectrum
Date: August 2023
Author: Arrykrishna Mootoovaloo
Collaborators: David Alonso and colleagues at Oxford and Imperial College.
"""
import numpy as np
import pyccl as ccl


COSMO = {"Omega_cdm": 0.25, "Omega_b": 0.04, "h": 0.70,
         "n_s": 1.0, "sigma8": 0.75}
AMEMU_TOLERANCE = 5e-2

PKLIN_CLASS = np.array(
    [
        1.17291900e03,
        1.64112151e03,
        2.29500349e03,
        3.20633895e03,
        4.47221195e03,
        6.22130738e03,
        8.61807981e03,
        1.18592776e04,
        1.61562285e04,
        2.16869285e04,
        2.85026512e04,
        3.63770036e04,
        4.45934428e04,
        5.17045353e04,
        5.53936266e04,
        5.30030340e04,
        4.39404404e04,
        3.30990480e04,
        2.61091983e04,
        1.73179613e04,
        1.11319778e04,
        6.29609558e03,
        3.57245034e03,
        1.90239549e03,
        9.67189818e02,
        4.74984720e02,
        2.26236298e02,
        1.05020635e02,
        4.76885905e01,
        2.12496264e01,
        9.31542019e00,
        4.02627736e00,
        1.71877641e00,
        7.25775853e-01,
        3.03509267e-01,
        1.25818251e-01,
        5.17389376e-02,
        2.11123552e-02,
        8.54673963e-03,
        3.42773963e-03,
    ],
    dtype=np.float32,
)


def test_amemu_linear():
    """
    Test the prediction for the linear matter power spectrum with the
    emulator, compared to CLASS.
    """
    emulator = ccl.amemuLinear(download=True)
    pklin_emu = emulator.get_pk_at_a(cosmo=COSMO, a=[1.0])
    err = np.abs((PKLIN_CLASS - pklin_emu) / PKLIN_CLASS)
    assert np.all(err < AMEMU_TOLERANCE)
