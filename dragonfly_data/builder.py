import pickle
import pandas as pd
from scipy.stats.mstats import winsorize


class MidPositionScaler:
    def __init__(self, position=0.1):
        self.position = position
        self.min_values = {}
        self.mid = {}

    def centering(self, X):
        for column in X.columns:
            if column not in self.min_values:
                min_val = X[column].min()
                self.min_values[column] = min_val
            else:
                min_val = self.min_values[column]
            X.loc[:, column] -= min_val
        return X

    def fit(self, X):
        X = self.centering(X)
        for column in X.columns:
            median = X[column].median()
            if median > 0:
                mid = median
            else:
                mid = X[column].mean()
                # mean = X[column].mean()
                # min_val = X[column][X[column] > 0].min()
                # if mean < min_val:
                #     mid = mean
                # else:
                #     mid = min_val
            self.mid[column] = mid

    def transform(self, X, cut=False):
        X = self.centering(X)
        for column in X.columns:
            scaling_factor = self.position / self.mid[column]
            X.loc[:, column] *= scaling_factor
            if cut:
                X.loc[X[column] > 1.0, column] = 1.0
        return X

    def fit_transform(self, X, cut=False):
        self.fit(X)
        return self.transform(X, cut)


class WinsorizeScaler:
    def __init__(self, limits=0.0001):
        self.limits = limits

    def transform(self, X):
        for column in X.columns:
            dtc = X[column].to_numpy().reshape(-1, 1)
            X.loc[:, column] = winsorize(dtc, limits=[self.limits, self.limits])
        return X


def preparer_datetime(dataframe,
                      column_time='DATETIME',
                      indexing=True):
    if dataframe[column_time].dtype == 'O':
        dataframe[column_time] = pd.to_datetime(dataframe[column_time])
    dataframe = dataframe.sort_values(by=column_time)
    if indexing:
        dataframe.set_index('DATETIME', inplace=True)
    return dataframe


def resample_datetime(dataframe,
                      timeframe,
                      fillna=False):
    """
    'S': Секунды; 'T' или 'min': Минуты; 'H': Часы; 'D': Дни; 'W': Недели;'M': Месяцы; 'Y': Годы
    '5T': 5 минут; '1H': 1 час; '2D': 2 дня; '3W': 3 недели; '6M': 6 месяцев; '10Y': 10 лет
    """

    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe.set_index('DATETIME', inplace=True)

    resampled_df = dataframe.resample(timeframe).agg({
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
        'TICKVOL': 'sum',
        'SPREAD': 'min'
    })

    if fillna:
        resampled_df = resampled_df.fillna(method='ffill')
    else:
        resampled_df = resampled_df.dropna()

    return resampled_df.reset_index()


def resample_tick(dataframe,
                  timeframe):
    """
    'S': Секунды; 'T' или 'min': Минуты; 'H': Часы; 'D': Дни; 'W': Недели;'M': Месяцы; 'Y': Годы
    '5T': 5 минут; '1H': 1 час; '2D': 2 дня; '3W': 3 недели; '6M': 6 месяцев; '10Y': 10 лет
    """

    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe.set_index('DATETIME', inplace=True)

    resampled_df = dataframe.rolling(timeframe).agg({
        'OPEN': lambda x: x.iloc[0],
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': lambda x: x.iloc[-1],
        'TICKVOL': 'sum',
        'SPREAD': 'min'
    })

    resampled_df = resampled_df.dropna()

    return resampled_df.reset_index()


def moving_average(dataframe,
                   window,
                   columns=None):
    if columns is None:
        columns = dataframe.columns
    for col in columns:
        dataframe[col] = dataframe[col].rolling(window=window).mean()
    return dataframe.dropna()
