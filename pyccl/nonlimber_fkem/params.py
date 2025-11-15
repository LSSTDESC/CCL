"""FFTLog parameter helpers for the FKEM non-Limber integration."""

__all__ = ["get_fftlog_params"]


def get_fftlog_params(
    bessel_index, nu_default=1.51, nu_low=0.51, plaw_default=0.0
):
    """Returns the FFTLog parameters: nu (bias), deriv (Bessel order), and plaw
    (power-law correction) for a given Bessel function index.

    This method uses default values from the FKEM paper (arXiv:1911.11947).

    Args:
        bessel_index (float or int):
            The order of the Bessel function.
        nu_default (float, optional):
            The default FFTLog bias parameter for non-negative Bessel indices.
            Defaults to 1.51.
        nu_low (float, optional):
            The FFTLog bias parameter for negative Bessel indices.
            Defaults to 0.51.
        plaw_default (float, optional):
            The default power-law correction for non-negative Bessel indices.
            Defaults to 0.0.

    Returns:
        tuple: A tuple containing:
            - nu (float): The FFTLog bias parameter.
            - deriv (float): The Bessel derivative order.
            - plaw (float): The power-law correction.

    Raises:
        TypeError: If `bessel_index` is not a real number.
        ValueError: If `bessel_index` is not finite or not an integer.
    """
    # Allow numpy scalars as input
    try:
        j = float(bessel_index)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"FKEM: bessel_index must be a real number, got {bessel_index!r}."
        ) from exc

    if not (j == j and abs(j) != float("inf")):  # NaN or inf
        raise ValueError("FKEM: bessel_index must be finite.")

    if int(round(j)) != j:
        raise ValueError(
            f"FKEM: bessel_index must be an integer, got {bessel_index!r}."
        )

    j = int(j)

    # Sanity on nu / plaw inputs
    for name, val in [
        ("nu_default", nu_default),
        ("nu_low", nu_low),
        ("plaw_default", plaw_default),
    ]:
        v = float(val)
        if not (v == v and abs(v) != float("inf")):
            raise ValueError(f"FKEM: {name} must be finite, got {val!r}.")

    if j < 0:
        # Negative-order case used in FKEM (e.g. for derivatives)
        return float(nu_low), 0.0, -2.0

    if j == 0:
        # Standard FKEM choice for j_0
        return float(nu_default), 0.0, float(plaw_default)

    # Positive integer j: use nu_default, deriv=j, plaw=0.0 (independent of
    # plaw_default)
    return float(nu_default), float(j), 0.0
