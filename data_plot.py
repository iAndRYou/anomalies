import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from data_handler import create_interval_column

def plot_volume(df: DataFrame) -> None:
    '''
    Plot the volume for BTC and USD
    '''
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x=df.index, y='Volume BTC', color='b', label='BTC volume')
    sns.lineplot(data=df, x=df.index, y='Volume USD', color='r', label='USD volume')

    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title(f'Volume for BTC and USD')

    plt.legend(title='Legend')
    plt.show()


def plot_price(df: DataFrame, interval_days_window: int = 150,
                outliers: DataFrame | None = None) -> None:
    '''
    Plot the close price and the means for intervals
    '''
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x=df.index, y='close', color='b', label='Price')
    
    # Add rolling mean to the plot
    hours_window = interval_days_window * 24
    sns.lineplot(data=df, x=df.index, y=df['close'].rolling(window=hours_window).mean(), color='g', label=f'{interval_days_window} days rolling mean')
    
    # Add interval means to the plot
    df = create_interval_column(df, interval_days_window)
    for interval in df['interval'].unique():
        mean = df.loc[df['interval'] == interval, 'close'].mean()
        xmin = df.loc[df['interval'] == interval].index.min()
        xmax = df.loc[df['interval'] == interval].index.max()
        sns.lineplot(x=[xmin, xmax], y=[mean, mean], color='r')
        
    if outliers is not None:
        sns.scatterplot(data=outliers, x=outliers.index, y='close', color='red', label='Price outliers', marker='o', s=100)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Price and means')

    plt.legend(title='Legend')
    plt.show()
    

def plot_mahalanobis(df: DataFrame, interval_days: int | None = None,
                     outliers: DataFrame | None = None) -> None:
    '''
    Plot the Mahalanobis distance for each interval
    '''
    sns.set_theme(style="whitegrid")
    
    if interval_days is None:
        sns.lineplot(data=df, x=df.index, y='MD')
        plt.title(f'Mahalanobis distance')
    else:
        # Plot for each interval
        df = create_interval_column(df, interval_days)
        for interval in df['interval'].unique():
            interval_data = df.loc[df['interval'] == interval]
            dates = interval_data.index
            distance = interval_data['MD']
            sns.lineplot(data=interval_data, x=dates, y=distance)
        
        plt.title(f'Mahalanobis distance for intervals')
    
    if outliers is not None:
        sns.scatterplot(data=outliers, x=outliers.index, y='MD', color='red', label='MD outliers', marker='o', s=100)

    plt.xlabel('Date')
    plt.ylabel('Mahalanobis distance')

    plt.show()
    

def plot_mahalanobis_distribution(df: DataFrame) -> None:
    '''
    Plot the distribution of the Mahalanobis distance
    '''
    sns.set_theme(style="whitegrid")
    ax = sns.histplot(data=df['MD'], kde=False)
    
    mean = df['MD'].mean()
    std = df['MD'].std()
    colors = ['b', 'g', 'r', 'k']
    
    for patch in ax.patches:
        for i in range(1, len(colors) + 1):
            if np.abs(patch.get_x() - mean) < (i * std):
                patch.set_facecolor(colors[i - 1])
                break

    plt.xlabel('Mahalanobis distance')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of the Mahalanobis distance')

    plt.show()
    
    
def plot_price_volume_corelation(df: DataFrame, hide_BTC: bool = False, hide_USD: bool = False, 
                                 outliers_BTC: DataFrame | None = None,
                                 outliers_USD: DataFrame | None = None, ) -> None:
    '''
    Plot the corelation between the price and the volume
    '''
    sns.set_theme(style="whitegrid")
    if not hide_USD:
        sns.scatterplot(data=df, x='close', y='Volume USD', color='r', label='USD volume')
    if not hide_BTC:
        sns.scatterplot(data=df, x='close', y='Volume BTC', color='b', label='BTC volume')
    if outliers_USD is not None:
        sns.scatterplot(data=outliers_USD, x='close', y='Volume USD', color='red', label='USD outliers', marker='o', s=100)
    if outliers_BTC is not None:
        sns.scatterplot(data=outliers_BTC, x='close', y='Volume BTC', color='blue', label='BTC outliers', marker='o', s=100)

    plt.xlabel('Price')
    plt.ylabel('Volume')
    plt.title(f'Corelation between price and volume')

    plt.legend(title='Legend')
    plt.show()
    
def plot_outliers(df: DataFrame, outliers: DataFrame, title: str) -> None:
    '''
    Plot the outliers for the close price
    '''
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x=df.index, y='close', color='b', label='Price')
        
    sns.scatterplot(data=outliers, x=outliers.index, y='close', color='red', label='Outliers', marker='o', s=100)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)

    plt.legend(title='Legend')
    plt.show()