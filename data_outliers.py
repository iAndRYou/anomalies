import numpy as np
import pandas as pd
from scipy import stats

def get_mahalanobis_outliers(df: pd.DataFrame, std_threshold: float = 4) -> pd.DataFrame:
    '''
    Get the mahalanobis outliers in the data
    '''
    df = df.copy()
    mean = df['MD'].mean()
    
    std = df['MD'].std()
    threshold = std_threshold * std
    
    # Identify outliers
    outliers = df[np.abs(df['MD'] - mean) > threshold]
    
    return outliers

def get_correlation_outliers(df: pd.DataFrame, data_column: str, std_threshold: float = 4) -> pd.DataFrame:
    '''
    Get the correlation outliers in the data
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

def get_merged_outliers(df: pd.DataFrame, data_column: str, std_threshold_mahalanobis: float = 4, std_threshold_correlation: float = 4) -> pd.DataFrame:
    '''
    Get the merged outliers for mahaanobis and given orrelation outliers in the data
    '''
    mahalanobis_outliers = get_mahalanobis_outliers(df, std_threshold_mahalanobis)
    correlation_outliers = get_correlation_outliers(df, data_column, std_threshold_correlation)
    
    # Get the intersection of the two outliers sets
    anomalies_index = pd.merge(mahalanobis_outliers, correlation_outliers, left_index=True, right_index=True, how='inner').index
    
    return df.loc[anomalies_index]