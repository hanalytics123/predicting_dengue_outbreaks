# Predicting Dengue Outbreaks Using Climate and Environmental Data

This repository contains the data processing, exploratory analysis, modelling and visualisation work for a data science project investigating whether **climate and environmental variables can be used to predict dengue outbreaks** using publicly available data.

### Project Overview

Dengue transmission is influenced by environmental conditions that affect *Aedes* mosquito survival, reproduction and breeding. Variables such as **temperature, rainfall and humidity** have consequently been widely investigated as predictors of dengue incidence.

This project combines historical dengue surveillance data with climate and environmental data to explore these relationships and develop predictive models. Multiple analytical approaches will be compared to determine which provides the most effective prediction of dengue incidence.

### Key Research Questions

- Can climate and environmental variables be used to predict dengue outbreaks?
- Which variables are most strongly associated with dengue incidence?
- Which modelling approach provides the most accurate predictions?
- Do machine learning approaches outperform traditional time-series methods?

### Data Sources

- **OpenDengue** – historical dengue case counts
- **ERA5** – temperature, humidity, wind speed and atmospheric conditions
- **NOAA PERSIANN** – precipitation data
- **NOAA NDVI** – vegetation data
- **NOAA GHCN-Daily** – ground-based weather observations

OpenDengue provides standardised publicly available dengue surveillance data covering more than 56 million cases across 102 countries.

### Planned Analysis

The project will include:

- Data cleaning, integration and feature engineering
- Exploratory data analysis and visualisation
- Analysis of temporal and lagged climate–dengue relationships
- Time-series and machine learning modelling
- Model comparison using metrics such as MAE and RMSE
- Feature importance and model interpretation

Candidate models include **SARIMA, Random Forest, Gradient Boosting and XGBoost**, reflecting approaches commonly used in dengue forecasting research.

### Tools & Technologies

- **Language:** Python
- **Data Analysis:** `pandas`, `numpy`
- **Visualisation:** `matplotlib`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Time Series:** `statsmodels`
- **Version Control:** Git & GitHub

### Repository Structure

```text
📁 data/           # Raw and processed datasets
📁 notebooks/      # Data preparation, EDA and modelling notebooks
📁 scripts/        # Reusable processing and modelling code
📁 outputs/        # Figures and model results
📄 README.md       # Project overview
```
### Project Status

*Work in progress*

The repository will be updated as data preparation, exploratory analysis, modelling and evaluation are completed.
