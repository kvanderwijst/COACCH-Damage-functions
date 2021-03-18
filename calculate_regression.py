import os
import pandas as pd
from PyPDF2 import PdfFileMerger, PdfFileReader

from regression import fitfunctions, quantreg, plot

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


#########
# Calculate regression coefficients
#########


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
    return {"values": pd.DataFrame(output).T, "x_param": x_param, "fit_fct": fct}


coeffs_no_slr_quadratic = coeffs(data_no_slr, "T_Delta", fitfunctions.Quadratic)
coeffs_slr_ad_linear = coeffs(data_slr_ad, "SLR", fitfunctions.Linear)
coeffs_slr_ad_logistic = coeffs(data_slr_ad, "SLR", fitfunctions.Logistic)
coeffs_slr_noad_linear = coeffs(data_slr_noad, "SLR", fitfunctions.Linear)
coeffs_slr_noad_quadratic = coeffs(data_slr_noad, "SLR", fitfunctions.Quadratic)


#########
# Save to Excel file
#########


def save(writer, calculated_coeffs, name):
    "Writes the coefficients to an Excel file"

    x_param = calculated_coeffs["x_param"]
    fit_fct = calculated_coeffs["fit_fct"]
    sheet_name = "{name}-{fctname}".format(name=name, fctname=fit_fct.__name__)

    # Put formula on first row
    fullformula = "a * ( {} )".format(fit_fct.formula.format(x=x_param))
    formula_df = pd.Series({"Formula": fullformula})
    formula_df.to_excel(writer, sheet_name=sheet_name, startrow=0, header=False)

    # Save coefficients to subsequent rows
    calculated_coeffs["values"].to_excel(writer, sheet_name=sheet_name, startrow=2)


with pd.ExcelWriter("damage_coefficients.xlsx") as writer:
    save(writer, coeffs_no_slr_quadratic, "NoSLR")
    save(writer, coeffs_slr_ad_linear, "SLR-Ad")
    save(writer, coeffs_slr_ad_logistic, "SLR-Ad")
    save(writer, coeffs_slr_noad_linear, "SLR-NoAd")
    save(writer, coeffs_slr_noad_quadratic, "SLR-NoAd")


#########
# Create Plotly figures
#########


def create_combined_plot_pdf(data, all_coeffs, outputname):
    filenames = []
    merged_pdf = PdfFileMerger()
    for i, region in enumerate(regions):
        filename = f"Output/temp_{i}.pdf"
        fig = plot.create_combined_plot(data, region, all_coeffs)
        fig.write_image(filename)
        filenames.append(filename)
        merged_pdf.append(PdfFileReader(filename))

    merged_pdf.write(outputname)

    # Delete all temporary files
    for file in filenames:
        os.remove(file)


create_combined_plot_pdf(
    data_no_slr, [coeffs_no_slr_quadratic], "TempRelatedDamages.pdf",
)
create_combined_plot_pdf(
    data_slr_ad, [coeffs_slr_ad_linear, coeffs_slr_ad_logistic], "SLRDamages_Ad.pdf",
)
create_combined_plot_pdf(
    data_slr_noad,
    [coeffs_slr_noad_linear, coeffs_slr_noad_quadratic],
    "SLRDamages_NoAd.pdf",
)

