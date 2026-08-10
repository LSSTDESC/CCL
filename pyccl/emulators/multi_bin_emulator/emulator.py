import numpy as np

# ------------------------------------------
# Native nonlinear grid
# ------------------------------------------

k_nl = np.load("/home/sankarshana/Codes/cola_emulation/k.npy")

# ------------------------------------------
# Instantiate sub-emulators
# ------------------------------------------

linear = LinearEmulator(
    "linear_boost_nn_multibin.pt"
)

nonlinear = NonLinearEmulator(
    "multibin_nn_emulator.pt",
    k_nl
)

# ------------------------------------------
# Combined emulator
# ------------------------------------------

emu = MGEmulator(
    linear,
    nonlinear
)
