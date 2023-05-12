from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import List

def incomplete_data_stores(sales_df: DataFrame) -> List[int]:
    """Identifies stores with incomplete timeline

    Args:
        sales_df (DataFrame): Dataframe containing sales data, must have columns:
        * sales_date: datetime
        * store: int

    Returns:
        List[int]: List of stores with incomplete timeline
    """
    stores=sales_df['store'].unique()
    list_of_incomplete_stores=[]
    for i in range(1, len(stores)+1):
        one_store=sales_df[sales_df['store']==i]
        if not pd.date_range(start = one_store['sales_date'].min(), end =one_store['sales_date'].max()).difference(one_store['sales_date']).empty:
            list_of_incomplete_stores.append(i)
    return list_of_incomplete_stores


def calendar_features(sales_df: DataFrame) -> DataFrame:
    """Generates date features

    Args:
        sales_df (DataFrame): Dataframe containing sales data, must have columns:
        * sales_date: datetime

    Returns:
        DataFrame: Original dataframe with additional date features
    """
    sales_df['year']=sales_df['sales_date'].dt.year
    sales_df['month']=sales_df['sales_date'].dt.month
    sales_df['day_of_month']=sales_df['sales_date'].dt.day
    sales_df['day_of_year']=sales_df['sales_date'].dt.dayofyear
    sales_df['week_of_year']=sales_df['sales_date'].dt.isocalendar().week
    sales_df['odd_weeks']=np.where(sales_df['week_of_year']%2==1,1,0)

    return sales_df


def features_to_categorical(df):
    # Convert StateHoliday to an ordered factor with levels "0", "a", "b", "c"
    df['state_holiday'] = pd.Categorical(df['state_holiday'], 
                                            categories=['0', 'a', 'b', 'c'], 
                                            ordered=True)


    # Create close and open_fct variables based on open
    df['close'] = 1 - df['open_flag']
    df['open_fct'] = pd.Categorical(df['open_flag'])

    # Create close_fct variable based on close
    df['close_fct'] = pd.Categorical(df['close'])

    return df


    
def create_date_features(df:DataFrame, date_column:str)->DataFrame:
    """Function to extract year, month, month_day, and week_day from date column of a dataframe

    Returns:
        the same dataframe with date features added
    """
    df['year'] = pd.to_datetime(df[date_column]).dt.year
    df['month'] = pd.to_datetime(df[date_column]).dt.month
    df['month_day'] = pd.to_datetime(df[date_column]).dt.day
    df['week_day'] = pd.to_datetime(df[date_column]).dt.day_name()

    # Convert week_day to an ordered factor with levels Monday, Tuesday, ..., Sunday
    df['week_day'] = pd.Categorical(df['week_day'], 
                                        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                    'Friday', 'Saturday', 'Sunday'], 
                                        ordered=True)
    
    return df