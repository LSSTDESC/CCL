__all__ = ("MuSigmaMG",)

from dataclasses import dataclass

from . import ModifiedGravity


@dataclass
class MuSigmaMG(ModifiedGravity):
    parametrisation: str = "mu_Sigma"
    mu_0: float = 0
    sigma_0: float = 0
    c1_mg: float = 1
    c2_mg: float = 1
    lambda_mg: float = 0
