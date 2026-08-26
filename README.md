## Climate-Enhanced Dengue Forecasting in Maynas, Loreto, Peru

This repository contains the code, data structures, exploratory analysis, forecasting models, and evaluation outputs used in a one-week-ahead dengue case forecasting project for **Maynas, Loreto, Peru**.

The project investigates whether climate and environmental variables can improve weekly dengue case prediction beyond recent dengue incidence alone. It compares simple forecasting baselines, traditional time-series models, and machine-learning approaches using a strict chronological evaluation design.

### Project Overview

Dengue is a climate-sensitive mosquito-borne disease whose transmission is influenced by rainfall, temperature, humidity, seasonality, and prior case burden. Reliable short-term forecasts can support public health planning by helping identify periods when dengue incidence may rise.

This project builds a weekly forecasting pipeline for dengue cases in Maynas using historical surveillance data combined with lagged and rolling climate/environmental predictors. The modelling task is deliberately framed as a **one-week-ahead regression problem**, where information available up to week `t-1` is used to predict dengue cases in week `t`.

The workflow includes exploratory data analysis, baseline forecasting, machine-learning model comparison, SARIMA/SARIMAX time-series modelling, and a final frozen evaluation on an untouched future holdout period. Results show that recent dengue incidence is the dominant forecasting signal, while climate/environmental variables provide a modest but measurable improvement in the final XGBoost model.

### Key Research Questions

- Can weekly dengue cases be forecast one week ahead using historical dengue incidence?
- Do climate and environmental predictors improve forecasting beyond dengue history and seasonality?
- How do machine-learning models compare with traditional SARIMA time-series models?
- Which models perform best during higher-incidence dengue weeks?
- Which predictors contribute most to the final climate-enhanced forecasting model?

### Data Sources

- **OpenDengue** – historical dengue case counts
- **ERA5** – temperature, humidity, wind speed and atmospheric conditions
- **NOAA PERSIANN** – precipitation data
- **NOAA NDVI** – vegetation data

### Tools & Technologies

- **Language:** Python
- **Core Packages:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `statsmodels`
- **Forecasting Methods:** Persistence baseline, Ridge regression, Random Forest, XGBoost, SARIMA, SARIMAX
- **Evaluation Metrics:** MAE, RMSE, R²
- **Development Environment:** Jupyter Notebook
- **Code Assistance:** ChatGPT - used for planning, code scaffolding, debugging, and language refinement

### Reproducibility

The modelling workflow uses a fixed chronological split:

- **Training:** 2000-2017
- **Validation:** 2018-2020
- **Final test:** 2021-2023

The final test period is held back from model selection and used only once model specifications are frozen. This prevents future information from influencing model development and provides a more realistic estimate of out-of-sample forecasting performance.

The engineered modelling dataset is designed for one-week-ahead prediction, with lagged dengue, seasonal, climate, and environmental features aligned so that predictors are available before the target week.

### Key Findings

| Model | Feature Set | Final Test MAE | Final Test RMSE | Final Test R² | Overall Result |
|---|---|---:|---:|---:|---|
| XGBoost | History + climate | **11.13** | **21.22** | **0.855** | Best final model |
| Persistence | Previous week's cases | 11.22 | 22.00 | 0.844 | Strong simple benchmark |
| XGBoost | History only | 11.45 | 22.35 | 0.839 | Did not beat persistence on final test |
| SARIMA | History + seasonality | 12.14 | 21.67 | 0.849 | Credible traditional comparator |

- **Recent dengue incidence is the strongest predictor** of one-week-ahead dengue cases.
- **Climate-enhanced XGBoost achieved the best final holdout performance**, but the improvement over persistence was modest.
- **Climate variables improved the final XGBoost model** by reducing MAE from 11.45 to 11.13 and RMSE from 22.35 to 21.22 compared with the same history-only XGBoost specification.
- **Validation results did not perfectly match final-test results**, highlighting the importance of an untouched future holdout period.
- **Higher-incidence weeks remained harder to predict**, although the climate-enhanced XGBoost model had the lowest MAE during these weeks.
- **Feature importance was dominated by recent dengue incidence**, especially the one-week lag, with smaller contributions from seasonality, precipitation, temperature, wind speed, and surface pressure.

### Repository Structure

```text
📁 data/           # Raw and processed datasets
📁 notebooks/      # Data preparation, EDA and modelling notebooks
📁 scripts/        # Reusable processing and modelling code
📁 outputs/        # Figures and model results
📄 README.md       # Project overview
```
### Project Status

*Completed*
