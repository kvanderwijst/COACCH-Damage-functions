import numpy as np
from statsmodels.api import QuantReg

from .fitfunctions import GeneralFitFunction


def multiplication_factor_quantiles(xvalues, yvalues, fit_fct: GeneralFitFunction):

    # First calculate best fit parameters for the function
    fit_params = fit_fct.fit(xvalues, yvalues)
    bestfit_params = fit_params["bestfit"]
    params_low = fit_params["low"]
    params_high = fit_params["high"]

    # Create the quantile regression model
    # (we round the bestfit_values to avoid QuantReg getting stuck in a local optimum)
    fit_values = fit_fct.fct(
        xvalues, *np.round(fit_fct.to_param_values(bestfit_params), 8)
    )
    ratios = yvalues / fit_values
    model = QuantReg(yvalues, fit_values)

    # Calculate the quantiles for each quantile
    quantiles = [0.025, 0.05, 0.16, 0.25, 0.33, 0.5, 0.67, 0.75, 0.84, 0.95, 0.975]
    factor_values = {
        f"a (q={q})": model.fit(q=q, p_tol=1e-12, max_iter=5000).params.values[0]
        for q in quantiles
    }

    all_outputs = {**bestfit_params, **factor_values}
    if params_low is not None:
        for key, value in params_low.items():
            all_outputs[f"{key}_low"] = value
    if params_high is not None:
        for key, value in params_high.items():
            all_outputs[f"{key}_high"] = value

    return all_outputs, ratios
