__all__ = ("MuSigmaMG",)

from dataclasses import dataclass

from . import ModifiedGravity


@dataclass
class MuSigmaMG(ModifiedGravity):
    """The mu-Sigma parametrization of modified gravity.

    Args:
        mu_0 (:obj:`float`): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        sigma_0 (:obj:`float`): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        c1_mg (:obj:`float`): MG parameter that enters in the scale
            dependence of mu affecting its large scale behavior. Default to 1.
            See, e.g., Eqs. (46) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        c2_mg (:obj:`float`): MG parameter that enters in the scale
            dependence of Sigma affecting its large scale behavior. Default 1.
            See, e.g., Eqs. (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        lambda_mg (:obj:`float`): MG parameter that sets the start
            of dependance on c1 and c2 MG parameters. Defaults to 0.0
            See, e.g., Eqs. (46) & (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
    """

    parametrization: str = "mu_Sigma"
    mu_0: float = 0
    sigma_0: float = 0
    c1_mg: float = 1
    c2_mg: float = 1
    lambda_mg: float = 0
