# COACCH-Damage-functions
Calculates the damage function coefficients for the COACCH project

In `calculate_regression.py`, define the data-folder. This folder should contain the following files:
 - `GDP_damages_IAMs_NoSLR_data.csv`
 - `GDP_damages_IAMs_SLR-Ad_data.csv`
 - `GDP_damages_IAMs_SLR-NoAd_data.csv`

All calculations happen in `regression/quantreg.py`. The available fit functions are defined in `regression/fitfunctions.py`, and currently include:
 - Linear
 - Linear with intercept
 - Quadratic
 - Logistic
 
## Requirements
This code uses OLS regression from `scipy.optimize` and quantile regression from `statsmodels`.

For the plotting part, we use the `plotly` package and the `pypdf2` package for combining the PDF files.
