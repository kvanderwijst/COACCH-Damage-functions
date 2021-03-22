"""
Define the possible functions for the best fit
"""

import numpy as np


class GeneralFitFunction:
    param_names = []
    formula = ""
    fct = lambda x, *b: None


class Linear(GeneralFitFunction):
    param_names = ["b1"]
    formula = "b1 * {x}"
    fct = lambda x, b1: b1 * x


class LinearWithIntercept(GeneralFitFunction):
    param_names = ["b1", "b2"]
    formula = "b1 + b2 * {x}"
    fct = lambda x, b1, b2: b1 + b2 * x


class Quadratic(GeneralFitFunction):
    param_names = ["b1", "b2"]
    formula = "b1 * {x} + b2 * {x}**2"
    fct = lambda x, b1, b2: b1 * x + b2 * x ** 2


class Logistic(GeneralFitFunction):
    param_names = ["b1", "b2", "b3"]
    formula = "b1/(1+b2*exp(-b3*{x})) - b1/(1+b2)"
    fct = lambda x, b1, b2, b3: b1 / (1 + b2 * np.exp(-b3 * x)) - b1 / (1 + b2)

