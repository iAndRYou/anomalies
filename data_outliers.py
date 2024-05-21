import numpy as np
from scipy import stats
from pandas import DataFrame

def get_mahalanobis_outliers(df: DataFrame, std_threshold: float = 4) -> DataFrame:
    '''
    Get the outliers in the data
    '''
    df = df.copy()
    mean = df['MD'].mean()
    
    std = df['MD'].std()
    threshold = std_threshold * std
    
    # Identify outliers
    outliers = df[np.abs(df['MD'] - mean) > threshold]
    
    return outliers

def get_correlation_outliers(df: DataFrame, data_column: str, std_threshold: float = 4) -> DataFrame:
    '''
    Get the outliers in the data
    '''
    df = df.copy()
    slope, intercept, _, _, _ = stats.linregress(df['close'], df[data_column])

    df['predicted_volume'] = intercept + slope * df['close']

    # Calculate residuals
    df['residuals'] = df[data_column] - df['predicted_volume']
    std_residuals = np.std(df['residuals'])

    threshold = std_threshold * std_residuals

    # Identify outliers
    outliers = df[np.abs(df['residuals']) > threshold]
    
    return outliers