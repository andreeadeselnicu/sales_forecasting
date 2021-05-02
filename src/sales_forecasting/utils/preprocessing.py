import pandas as pd
import numpy as np
from pandas import DataFrame
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
