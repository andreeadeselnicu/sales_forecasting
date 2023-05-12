import pandas as pd
import numpy as np
from datetime import datetime
from numpy import ndarray
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

models=[]
daily_smape=[]
weekly_smape=[]
monthly_smape=[]
percentage_error=[]
list_of_incomplete_stores=[]
store=[]


def train_test_validate_split_on_date(df, validate_begin:datetime, validate_end:datetime, datetime_col: str,output_col:str ):
    train_data = df[df[datetime_col] < validate_begin].copy()
    validate_data = df[(df[datetime_col] >= validate_begin) & (df[datetime_col] < validate_end)].copy()
    test_data = df[df[datetime_col] >= validate_end].copy()

    x_train = train_data.drop(columns=[datetime_col,output_col])
    y_train = train_data[output_col]

    x_test = test_data.drop(columns=[datetime_col,output_col])
    y_test = test_data[output_col]

    x_validate = validate_data.drop(columns=[datetime_col,output_col])
    y_validate = validate_data[output_col]

    return x_train, y_train, x_test, y_test, x_validate, y_validate


def predicted_vs_real_plot(y_test, y_pred, title: str, image_name: str):
    plt.figure(figsize=(8,5))  
    plt.plot(y_test.values, color='blue', label='Real sales values')  
    plt.plot(y_pred , color='red', label='Predicted sales values')  
    plt.title(title)  
    plt.xlabel('data')  
    plt.ylabel('Sales quantity')  
    plt.legend()
    plt.savefig(f"{image_name}.png")  
    plt.show()


def results_df(y_test:Series, y_hat:ndarray)->DataFrame:
    """Creates result dataframe with real sales and predictions on daily, weekly and monthly levels

    Args:
        y_test (Series): real sales values
        y_hat (ndarray): predicted sales values

    Returns:
        DataFrame: dataframe with real values, predicted ones, week and month aggregations
    """
    results = pd.DataFrame()
    results['real_sales'] = y_test
    results['predicted_sales'] = y_hat.round().astype('int')
    results.reset_index(inplace=True, drop=True)
    results['week_group']=(results.index/7).astype(int)
    results['month_group']=(results.index/30).astype(int)

    return results


#function for keeping the model results
def results_keeper( model_name, daily_error_smape:float, weekly_error_smape:float, monthly_error_smape:float, percentage_error_model:float):
    """Creates dataframe with used model info and different metrics

    Args:
        model_name ([type]): machine learning model class
        daily_error_smape (float): smape error calculated at daily level
        weekly_error_smape (float): smape error calculated at weekly level
        monthly_error_smape (float): smape error calculated at monthly level
        percentage_error_model (float): percentage error at daily level

    Returns:
        lists with info: infos are about models_name and calculated errors
    """
    model_str=str(model_name)
    if "(" in model_str:
        model_str=model_str.split("(")[0]
    elif "." in model_str:
        model_str=model_str.split(".")[0]
        model_str=model_str.split("<")[1]
    models.append(model_str)
    daily_smape.append(round(daily_error_smape,2))
    weekly_smape.append(round(weekly_error_smape,2))
    monthly_smape.append(round(monthly_error_smape,2))
    percentage_error.append(percentage_error_model)
    return models, daily_smape, weekly_smape, monthly_smape, percentage_error


#function for keeping the model results
def final_results_keeper( model_name, daily_error_smape:float, weekly_error_smape:float, monthly_error_smape:float, percentage_error_model:float, store_value:int):
    """Result keeper for all stores models

    Args:
        model_name ([type]): machine learning model class
        daily_error_smape (float): smape error calculated at daily level
        weekly_error_smape (float): smape error calculated at weekly level
        monthly_error_smape (float): smape error calculated at monthly level
        percentage_error_model (float): percentage error at daily level
        store_value (int): int value representing store number
    Returns:
        lists with info: infos are about models_name, calculated errors corresponding store
    """
    store.append(store_value)
    daily_smape.append(round(daily_error_smape,2))
    weekly_smape.append(round(weekly_error_smape,2))
    monthly_smape.append(round(monthly_error_smape,2))
    percentage_error.append(percentage_error_model)
    return models, daily_smape, weekly_smape, monthly_smape, percentage_error, store


def prediction_metrics_and_result_keeper(y_test:Series,y_pred:ndarray, model_name):
    """Creates dataframe with used model info and different metrics for exploratory use with metrics values printed

    Args:
        y_test (Series): real sales values
        y_pred (ndarray): predicted sales values
        model_name ([type]): machine learning model class description

    Returns:
        lists with info: infos are about models_name and calculated errors 
    """
    
    predictions=results_df(y_test=y_test, y_hat=y_pred)
    weekly_metrics=predictions.groupby(['week_group']).sum()
    monthly_metrics=predictions.groupby(['month_group']).sum() 

    #Get daily, weekly and monthly error values

    daily_error_smape=smape(predictions['real_sales'], predictions['predicted_sales'])
    weekly_error_smape=smape(weekly_metrics['real_sales'],weekly_metrics['predicted_sales'])
    monthly_error_smape=smape(monthly_metrics['real_sales'],monthly_metrics['predicted_sales'])
    percentage_error_rf=round(np.abs(predictions["real_sales"].sum()-predictions["predicted_sales"].sum())/predictions["real_sales"].sum()*100,2)

    print('Total value of sales on test predicted interval:', predictions['real_sales'].sum())
    print('Absolute error: ', sum(np.abs(error(actual=predictions['real_sales'], predicted=predictions['predicted_sales']))))
    print(f'Percentage error reported to real sales: {percentage_error_rf}%')
    print(f"Daily SMAPE is: {round(daily_error_smape,2)} %")
    print(f"Weekly SMAPE is: { round(weekly_error_smape,2)} %")
    print(f"Monthly SMAPE is: {round(monthly_error_smape,2)} %")
    # Add results to results keeper df
    models, daily_smape, weekly_smape, monthly_smape, percentage_error_model = results_keeper(model_name,daily_error_smape,weekly_error_smape, monthly_error_smape,percentage_error_rf)
    return models, daily_smape, weekly_smape, monthly_smape, percentage_error_model


def store_train_test_selection(sales_df:DataFrame, store_number:int):
    """Create x and y train-test based on selected store number

    Args:
        sales_df (DataFrame): dataframe with all sales data
        store_number (int): store number intended to be selected

    Returns:
        dataframes with x - y train-test based on selected store number
    """

    one_store=sales_df[sales_df['store']==store_number].copy()
    train_data=one_store[one_store['sales_date']<datetime(2015,2,28)].copy()
    test_data=one_store[one_store['sales_date']>=datetime(2015,2,28)].copy()
    x_train=train_data.drop(columns=['sales_date','store','sales','customers','state_holiday','close_flag'])
    y_train=train_data['sales']
    x_test=test_data.drop(columns=['sales_date','store','sales','customers','state_holiday','close_flag'])
    y_test=test_data['sales'] 

    return x_train, y_train, x_test, y_test


def model_selection_specific_store(random_forest_model_current_store, y_test:Series, y_pred_rf:ndarray, model_selection_nonstop_stores:str, store_number)->DataFrame:

    predictions=results_df(y_test=y_test, y_hat=y_pred_rf)
    weekly_metrics=predictions.groupby(['week_group']).sum()
    monthly_metrics=predictions.groupby(['month_group']).sum() 

    #Get daily, weekly and monthly error values

    daily_error_smape=smape(predictions['real_sales'], predictions['predicted_sales'])
    weekly_error_smape=smape(weekly_metrics['real_sales'],weekly_metrics['predicted_sales'])
    monthly_error_smape=smape(monthly_metrics['real_sales'],monthly_metrics['predicted_sales'])
    percentage_error_rf=round(np.abs(predictions["real_sales"].sum()-predictions["predicted_sales"].sum())/predictions["real_sales"].sum()*100,2)

    models, daily_smape, weekly_smape, monthly_smape, percentage_error, store = final_results_keeper(random_forest_model_current_store, daily_error_smape, weekly_error_smape, monthly_error_smape,
     percentage_error_rf, store_value=store_number)

    model_selection_nonstop_stores = pd.DataFrame({'Store': store, 'daily_smape': daily_smape, 'weekly_smape': weekly_smape, 'monthly_smape' :monthly_smape,'percentage_error':     percentage_error})
    
    return model_selection_nonstop_stores


def error(actual, predicted):
    return actual - predicted

def smape(real, predicted):
    return (1/len(real)) * np.sum(np.abs(real - predicted) / ((np.abs(real) + np.abs(predicted)) / 2)) * 100