import numpy as np
from statsmodels.api import QuantReg
from scipy.optimize import curve_fit

from .fitfunctions import GeneralFitFunction


def multiplication_factor_quantiles(xvalues, yvalues, fit_fct: GeneralFitFunction):

    # First calculate best fit parameters for the function
    bestfit_values, _ = curve_fit(fit_fct.fct, xvalues, yvalues)

    # Combine the bestfit values with the parameter names
    bestfit_params = dict(zip(fit_fct.param_names, bestfit_values))

    # Create the quantile regression model
    # (we round the bestfit_values to avoid QuantReg getting stuck in a local optimum)
    model = QuantReg(yvalues, fit_fct.fct(xvalues, *np.round(bestfit_values, 5)))

    # Calculate the quantiles for each quantile
    quantiles = [0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975]
    factor_values = {
        f"a (q={q})": model.fit(q=q, p_tol=1e-10, max_iter=5000).params.values[0]
        for q in quantiles
    }

    return {**bestfit_params, **factor_values}
