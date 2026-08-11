import numpy as np


class MGEmulator:

    def __init__(
        self,
        linear_emu,
        nonlinear_emu,
        k_low=0.5e-2,
        k_high=2e-2,
        k_internal=None
    ):

        self.linear = linear_emu
        self.nonlinear = nonlinear_emu

        self.k_low = k_low
        self.k_high = k_high

        if k_internal is None:

            self.k_internal = np.logspace(
                -4,
                1,
                2000
            )

        else:

            self.k_internal = np.asarray(
                k_internal
            )

        self.w = self._compute_weights(
            self.k_internal
        )[None, :]

    # ==================================================
    # Blending weights
    # ==================================================

    def _compute_weights(self, k):

        logk = np.log(k)

        center = 0.5 * (
            np.log(self.k_low)
            + np.log(self.k_high)
        )

        width = (
            np.log(self.k_high)
            - np.log(self.k_low)
        ) / 5.0

        w = 0.5 * (
            1.0 + np.tanh(
                (logk - center) / width
            )
        )

        return w

    # ==================================================
    # Main prediction
    # ==================================================

    def predict(
        self,
        zs,
        omega_m,
        omega_b,
        h,
        n_s,
        A_s,
        mus,
        etas,
        k=None
    ):

        zs = np.atleast_1d(zs)

        # --------------------------------------
        # Linear emulator
        # --------------------------------------

        boost_lin = np.array([

            self.linear.predict(
                omega_m=omega_m,
                omega_b=omega_b,
                h=h,
                n_s=n_s,
                ln1e10As=A_s,
                mus=mus,
                etas=etas,
                z_pk=z,
                k=self.k_internal
            )

            for z in zs

        ])

        # --------------------------------------
        # Nonlinear emulator
        # --------------------------------------

        boost_nl = self.nonlinear.predict(
            zs=zs,
            omega_m=omega_m,
            omega_b=omega_b,
            h=h,
            n_s=n_s,
            A_s=A_s,
            mus=mus,
            k=self.k_internal
        )

        # --------------------------------------
        # Stitch
        # --------------------------------------

        boost = (
            (1.0 - self.w) * boost_lin
            +
            self.w * boost_nl
        )

        # --------------------------------------
        # Return internal grid
        # --------------------------------------

        if k is None:

            return self.k_internal, boost

        # --------------------------------------
        # Interpolate to user grid
        # --------------------------------------

        k = np.asarray(k)

        boost_interp = np.empty(
            (len(zs), len(k))
        )

        for i in range(len(zs)):

            boost_interp[i] = np.exp(

                np.interp(
                    np.log(k),
                    np.log(self.k_internal),
                    np.log(boost[i])
                )

            )

        return k, boost_interp
