# Sales Forecasting

## 1. Project description:
The aim of this project is to build a forecasting model that predicts sales for a single product comercialized in 1115 stores. The end results will be used in the supply chain process and for the projection of the monthly revenue.
The data covers 2.5 years of daily sales. Also, the dataset contains information about promotions, the daily number of customers, and 2 types of holidays (school and state).

## 2. Solution:

### Project structure:
- Understanding, cleaning, and exploring data (`notebooks/exploratory_data_analysis.ipynb`);
- Data modeling & validation (`src/sales_forecasting/api/model.py`)
- Feature engineering (`src/sales_forecasting/utils/feature_engineering.py`);
- Experimenting with machine learning models (LightGBMRegressor, Random Forest) (`notebooks/machine_learning.ipynb`)
- Choosing the final models and evaluation (`notebooks/machine_learning.ipynb`, `src/sales_forecasting/utils/ml.py`)

### Tools used:
- Python 3.10
- Poetry for packaging and version management
- Pydantic for data modeling and validation
- MLFlow for experiments tracking 

### Metric:
- The metric used for evaluation is `Symmetric Mean Absolute Percentage Error (SMAPE)` at daily, weekly and monthly granularities.

### Implementation particularities:
 After the data exploration and modeling, I have decided to create 3 machine learning models using the `Random Forest Regressor` algorithm. The models will be created for different type of stores: non-stop stores, normal stores and stores that contain gaps in data.


## How to use:
```
pip install poetry
poetry install # will install sales_forecasting as python package with all dependencies
poetry shell # to activate the virtual environment
# and notebooks are ready to run
```