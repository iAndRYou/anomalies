import pandas as pd
import numpy as np
from data.metrics import mahalanobis_distance

def get_bitcoin_data() -> pd.DataFrame:
    '''
    Load the Bitcoin hourly data from the CSV file and return it as a DataFrame.
    '''
    df = pd.read_csv('./data/BTC-Hourly.csv')
    df = df.drop(columns=['date', 'symbol'])
    
    # Replace and drop NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df


def index_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Set the 'datetime' column as the new index and return the DataFrame.
    '''
    df['datetime'] = pd.to_datetime(df['unix'], unit='s')
    df = df.set_index('datetime')
    return df


def create_interval_column(df: pd.DataFrame, interval_days: int = 150) -> pd.DataFrame:
    '''
    Create a new column 'interval' based on the 'datetime' column and the interval_days.
    '''
    df['datetime'] = pd.to_datetime(df['unix'], unit='s')
    start_date = df['datetime'].min()

    df['interval'] = ((df['datetime'] - start_date).dt.days // interval_days) + 1

    return df


def apply_mahalanobis_interval(df: pd.DataFrame, interval_days: int = 150) -> pd.DataFrame:
    '''
    Apply the Mahalanobis distance based on days intervals to each row in the DataFrame and return it.
    '''
    df = create_interval_column(df, interval_days)

    # Calculate Mahalanobis distance for each interval
    for interval in df['interval'].unique():
        # Get 2D data [timestamp, close price]
        unix = df.loc[df['interval'] == interval, 'unix'].to_numpy()
        close = df.loc[df['interval'] == interval, 'close'].to_numpy()
        zip_data = np.column_stack((unix, close))
        
        # Apply Mahalanobis distance to each row
        df.loc[df['interval'] == interval, 'MD'] = \
            df.loc[df['interval'] == interval].apply(lambda x: 
                mahalanobis_distance(zip_data, [x['unix'], x['close']]), axis=1)
    
    return df

def add_anomaly_column(df: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    '''
    Add a new column 'anomaly' to the DataFrame based on the outliers.
    '''
    df['anomaly'] = False
    df.loc[outliers.index, 'anomaly'] = True
    
    return df