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


def plot_price(df: DataFrame, interval_days_window: int = 150) -> None:
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
    

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Price and means')

    plt.legend(title='Legend')
    plt.show()
    

def plot_mahalanobis_intervals(df: DataFrame, interval_days: int = 150) -> None:
    '''
    Plot the Mahalanobis distance for each interval
    '''
    sns.set_theme(style="whitegrid")
    
    # Plot for each interval
    df = create_interval_column(df, interval_days)
    for interval in df['interval'].unique():
        interval_data = df.loc[df['interval'] == interval]
        dates = interval_data.index
        distance = interval_data['MD']
        sns.lineplot(data=interval_data, x=dates, y=distance)

    plt.xlabel('Date')
    plt.ylabel('Mahalanobis distance')
    plt.title(f'Mahalanobis distance for intervals')

    plt.show()