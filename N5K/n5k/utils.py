from .calculator_base import N5KCalculatorBase


def n5k_calculator_from_name(name):
    """ Returns mass function subclass from name string
    Args:
        name (string): a mass function name
    Returns:
        MassFunc subclass corresponding to the input name.
    """
    calculators = {c.name: c for c in N5KCalculatorBase.__subclasses__()}
    if name in calculators:
        return calculators[name]
    else:
        raise ValueError("Calculator %s not implemented" % name)
