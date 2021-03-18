import numpy as np
from statsmodels.api import QuantReg
from scipy.optimize import curve_fit, least_squares

from .fitfunctions import GeneralFitFunction


def multiplication_factor_quantiles(
    xvalues, yvalues, fit_fct: GeneralFitFunction, robust_regression=False
):

    # First calculate best fit parameters for the function
    if robust_regression:
        # Robust regression uses the soft L1 norm, using the `least_squares` function from scipy
        def fct_mapping(params, x, y):
            return fit_fct.fct(x, *params) - y

        res_robust = least_squares(
            fct_mapping,
            np.ones(len(fit_fct.param_names)),
            args=(xvalues, yvalues),
            loss="soft_l1",
            f_scale=0.01,
        )
        bestfit_values = res_robust.x
    else:
        bestfit_values, _ = curve_fit(fit_fct.fct, xvalues, yvalues)

    # Combine the bestfit values with the parameter names
    bestfit_params = dict(zip(fit_fct.param_names, bestfit_values))

    # Create the quantile regression model
    # (we round the bestfit_values to avoid QuantReg getting stuck in a local optimum)
    model = QuantReg(yvalues, fit_fct.fct(xvalues, *np.round(bestfit_values, 8)))

    # Calculate the quantiles for each quantile
    quantiles = [0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975]
    factor_values = {
        f"a (q={q})": model.fit(q=q, p_tol=1e-12, max_iter=5000).params.values[0]
        for q in quantiles
    }

    return {**bestfit_params, **factor_values}
