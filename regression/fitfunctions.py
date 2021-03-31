"""
Define the possible functions for the best fit
"""

from abc import abstractclassmethod
from typing import Callable
import numpy as np
from scipy.optimize import least_squares, curve_fit
import statsmodels.formula.api as smf


class GeneralFitFunction:
    param_names = []
    formula = ""
    fct = lambda x, *b: None

    @abstractclassmethod
    def fit(cls, xvalues, yvalues):
        pass

    @classmethod
    def to_param_dict(cls, param_values):
        assert len(param_values) == len(
            cls.param_names
        ), "Number of parameters doesn't match"
        return dict(zip(cls.param_names, param_values))

    @classmethod
    def to_param_values(cls, param_dict):
        # We could have just used param_dict.values(), but we cannot be sure
        # that the dictionary is ordered
        return np.array([param_dict[param] for param in cls.param_names])


class GeneralStatsmodelsFitFunction(GeneralFitFunction):

    patsy_formula = "y ~ x"
    sm_model_fct: Callable

    @classmethod
    def fit(cls, xvalues, yvalues):
        model = cls.sm_model_fct(cls.patsy_formula, {"x": xvalues, "y": yvalues})
        result = model.fit()

        params_bestfit = result.params.values
        params_p_low = result.conf_int()[0].values
        params_p_high = result.conf_int()[1].values

        return {
            "bestfit": cls.to_param_dict(params_bestfit),
            "low": cls.to_param_dict(params_p_low),
            "high": cls.to_param_dict(params_p_high),
        }


class GeneralScipyRobustLS(GeneralFitFunction):
    method = "LS_soft_l1"
    loss = "soft_l1"
    f_scale = 0.01

    @classmethod
    def fit(cls, xvalues, yvalues):

        # Scipy uses a different functional form:
        def fct_mapping(params, x, y):
            return cls.fct(x, *params) - y

        # Robust regression uses the soft L1 norm, using the `least_squares`
        # function from scipy
        result = least_squares(
            fct_mapping,
            np.ones(len(cls.param_names)),
            args=(xvalues, yvalues),
            loss=cls.loss,
            f_scale=cls.f_scale,
        )

        params_bestfit = result.x

        return {
            "bestfit": cls.to_param_dict(params_bestfit),
            "low": None,
            "high": None,
        }


class GeneralScipyOLS(GeneralFitFunction):
    method = "OLS"

    @classmethod
    def fit(cls, xvalues, yvalues):
        params_bestfit, _ = curve_fit(cls.fct, xvalues, yvalues)

        return {
            "bestfit": cls.to_param_dict(params_bestfit),
            "low": None,
            "high": None,
        }


class GeneralLinear(GeneralStatsmodelsFitFunction):
    param_names = ["b1"]
    formula = "b1 * {x}"
    fct = lambda x, b1: b1 * x
    patsy_formula = "y ~ x - 1"


class GeneralQuadratic(GeneralStatsmodelsFitFunction):
    param_names = ["b1", "b2"]
    formula = "b1 * {x} + b2 * {x}"
    fct = lambda x, b1, b2: b1 * x + b2 * x ** 2
    patsy_formula = "y ~ x + I(x**2) - 1"


class LinearOLS(GeneralLinear):
    method = "OLS"
    sm_model_fct = smf.ols


class LinearQuantReg(GeneralLinear):
    method = "quant_reg"
    sm_model_fct = smf.quantreg


class QuadraticOLS(GeneralQuadratic):
    method = "OLS"
    sm_model_fct = smf.ols


class QuadraticQuantReg(GeneralQuadratic):
    method = "quant_reg"
    sm_model_fct = smf.quantreg


class LogisticRobust(GeneralScipyRobustLS):
    param_names = ["b1", "b2", "b3"]
    formula = "b1/(1+b2*exp(-b3*{x})) - b1/(1+b2)"
    fct = lambda x, b1, b2, b3: b1 / (1 + b2 * np.exp(-b3 * x)) - b1 / (1 + b2)


class LogisticOLS(GeneralScipyOLS):
    param_names = ["b1", "b2", "b3"]
    formula = "b1/(1+b2*exp(-b3*{x})) - b1/(1+b2)"
    fct = lambda x, b1, b2, b3: b1 / (1 + b2 * np.exp(-b3 * x)) - b1 / (1 + b2)
