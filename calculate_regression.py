import pandas as pd

from regression import fitfunctions, quantreg

# Import data
folder = "Data 2021-03-07/"
data_no_slr = pd.read_csv(folder + "GDP_damages_IAMs_NoSLR_data.csv")
data_slr_ad = pd.read_csv(folder + "GDP_damages_IAMs_SLR-Ad_data.csv")
data_slr_noad = pd.read_csv(folder + "GDP_damages_IAMs_SLR-NoAd_data.csv")

# Filter data
data_no_slr = data_no_slr[data_no_slr["d_T_Delta"] >= 0.05]
data_slr_ad = data_slr_ad[data_slr_ad["level"] != "High-end"]
data_slr_noad = data_slr_noad[data_slr_noad["level"] != "High-end"]

# Get all the regions
regions_nonworld = (
    data_no_slr.loc[data_no_slr["Region"] != "World", "Region"].sort_values().unique()
)
regions = ["World"] + list(regions_nonworld)


def coeffs(data, x_param, fct):
    """
    For each region, calculates the best fit parameter values
    corresponding to the fit function, and the multiplicatio
    factors from the quantile regression
    """
    output = {}
    for region in regions:
        subset = data[data["Region"] == region]
        output[region] = quantreg.multiplication_factor_quantiles(
            subset[x_param], subset["Value"], fct
        )
    return pd.DataFrame(output).T


def save(writer, data, x_param, fit_fct, name):
    "Writes the coefficients to an Excel file"

    calculated_coeffs = coeffs(data, x_param, fit_fct)
    sheet_name = "{name}-{fctname}".format(name=name, fctname=fit_fct.__name__)

    # Put formula on first row
    fullformula = "a * ( {} )".format(fit_fct.formula.format(x=x_param))
    formula_df = pd.Series({"Formula": fullformula})
    formula_df.to_excel(writer, sheet_name=sheet_name, startrow=0, header=False)

    # Save coefficients to subsequent rows
    calculated_coeffs.to_excel(writer, sheet_name=sheet_name, startrow=2)


# Create output file
with pd.ExcelWriter("damage_coefficients.xlsx") as writer:

    save(writer, data_no_slr, "T_Delta", fitfunctions.Quadratic, "NoSLR")
    save(writer, data_no_slr, "T_Delta", fitfunctions.Quadratic, "NoSLR")
    save(writer, data_slr_ad, "SLR", fitfunctions.Linear, "SLR-Ad")
    save(writer, data_slr_ad, "SLR", fitfunctions.Logistic, "SLR-Ad")
    save(writer, data_slr_noad, "SLR", fitfunctions.Linear, "SLR-NoAd")
    save(writer, data_slr_noad, "SLR", fitfunctions.Quadratic, "SLR-NoAd")

