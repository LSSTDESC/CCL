__all__ = ("MuSigmaMG",)

from dataclasses import dataclass

from . import ModifiedGravity


@dataclass
class MuSigmaMG(ModifiedGravity):
    """The :math:`\\mu`-:math:`\\Sigma` parametrization of modified gravity.

    Args:
        mu_0 (:obj:`float`): One of the parameters of the
            :math:`\\mu`-:math:`\\Sigma`
            modified gravity model. Defaults to 0.0
        sigma_0 (:obj:`float`): One of the parameters of the
            :math:`\\mu`-:math:`\\Sigma`
            modified gravity model. Defaults to 0.0
        c1_mg (:obj:`float`): MG parameter that enters in the scale
            dependence of :math:`\\mu`, affecting its large scale behavior.
            Default to 1. See, e.g., Eqs. (46) in
            `Ade et al. 2015 <https://arxiv.org/abs/1502.01590>`_,
            where their :math:`f_1` and :math:`f_2` functions are set equal
            to the commonly used ratio of dark energy density parameter at
            scale factor a over the dark energy density parameter today
        c2_mg (:obj:`float`): MG parameter that enters in the scale
            dependence of :math:`\\Sigma` affecting its large scale behavior.
            Default 1. See, e.g., Eqs. (47) in
            `Ade et al. 2015 <https://arxiv.org/abs/1502.01590>`_,
            where their :math:`f_1` and :math:`f_2` functions are set equal
            to the commonly used ratio of dark energy density parameter at
            scale factor a over the dark energy density parameter today
        lambda_mg (:obj:`float`): MG parameter that sets the start
            of dependence on :math:`c_1` and :math:`c_2` MG parameters.
            Defaults to 0.0 See, e.g., Eqs. (46) & (47) in
            `Ade et al. 2015 <https://arxiv.org/abs/1502.01590>`_,
            where their :math:`f_1` and :math:`f_2` functions are set equal
            to the commonly used ratio of dark energy density parameter at
            scale factor a over the dark energy density parameter today
    """

    parametrization: str = "mu_Sigma"
    mu_0: float = 0
    sigma_0: float = 0
    c1_mg: float = 1
    c2_mg: float = 1
    lambda_mg: float = 0
