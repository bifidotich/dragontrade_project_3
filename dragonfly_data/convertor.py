import numpy as np
import pandas as pd
from utils.files import track_dir
from sklearn.preprocessing import MinMaxScaler
from ta.trend import adx, adx_neg, adx_pos, macd, sma_indicator, ema_indicator, stc


def to_dragonfly_format(dataframe,
                        path_save=None,
                        indicators=True,
                        window_smooth=0,
                        drop_last_bar=True,
                        num_time=0):

    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe.sort_values(by='DATETIME', inplace=True)

    if drop_last_bar:
        dataframe.drop(dataframe.tail(1).index, inplace=True)

    if num_time > 0:
        dataframe['TIMENUMERIC'] = (pd.to_datetime(dataframe['DATETIME']) - pd.to_datetime(
            dataframe['DATETIME']).dt.normalize()).dt.total_seconds()
        dataframe['TIMENUMERIC'] = dataframe['TIMENUMERIC'] // (60 * 60 * 24 // num_time)

    if indicators:
        dataframe['ADX'] = adx(dataframe['HIGH'], dataframe['LOW'], dataframe['CLOSE'], fillna=True)
        dataframe['ADX_POS'] = adx_pos(dataframe['HIGH'], dataframe['LOW'], dataframe['CLOSE'], fillna=True)
        dataframe['ADX_NEG'] = adx_neg(dataframe['HIGH'], dataframe['LOW'], dataframe['CLOSE'], fillna=True)
        dataframe['MACD'] = macd(dataframe['CLOSE'], fillna=True)
        # dataframe['MACD-MM'] = MinMaxScaler().fit_transform(dataframe[['MACD']])
        # dataframe['MACD-UP'] = (dataframe['MACD']).apply(lambda x: x if x >= 0 else 0)
        # dataframe['MACD-DOWN'] = (dataframe['MACD']).apply(lambda x: abs(x) if x < 0 else 0)
        dataframe['SMA'] = sma_indicator(dataframe['CLOSE'], fillna=True)
        dataframe['EMA'] = ema_indicator(dataframe['CLOSE'], window=200, fillna=True)
        dataframe['STC'] = stc(dataframe['CLOSE'], fillna=True)

    if window_smooth > 1:
        smooth_cols = ['OPEN', 'CLOSE', 'HIGH', 'LOW']
        smoothed_numeric_cols = dataframe[smooth_cols].rolling(window=window_smooth, center=False, min_periods=1).mean()
        dataframe[smoothed_numeric_cols.columns] = smoothed_numeric_cols

    dataframe['OPEN-HIGH'] = dataframe['HIGH'] - dataframe['OPEN']
    dataframe['OPEN-LOW'] = dataframe['OPEN'] - dataframe['LOW']
    dataframe['OPEN-CLOSE'] = dataframe['CLOSE'] - dataframe['OPEN']
    dataframe['OPEN-CLOSE-DOWN'] = (dataframe['OPEN'] - dataframe['CLOSE']).apply(lambda x: max(0, x))
    dataframe['OPEN-CLOSE-UP'] = (dataframe['CLOSE'] - dataframe['OPEN']).apply(lambda x: max(0, x))

    if 'SPREAD' not in dataframe.columns:
        dataframe['SPREAD'] = dataframe['HIGH'] - dataframe['LOW']

    for column in ['TICKVOL', 'SPREAD']:
        col = dataframe.pop(column)
        dataframe[column] = col

    dataframe = dataframe.fillna(0.0)

    if path_save:
        track_dir(path_save)
        dataframe.to_csv(path_save, index=False)

    return dataframe


def from_MT5Terminal(dataframe,
                     path_save=None):
    dataframe.rename(columns={'<OPEN>': 'OPEN', '<HIGH>': 'HIGH', '<LOW>': 'LOW', '<CLOSE>': 'CLOSE'}, inplace=True)
    dataframe.rename(columns={'<TICKVOL>': 'TICKVOL', '<SPREAD>': 'SPREAD'}, inplace=True)
    dataframe['DATETIME'] = pd.to_datetime(dataframe['<DATE>'] + ' ' + dataframe['<TIME>'])
    dataframe = dataframe.drop(['<DATE>', '<TIME>', '<VOL>'], axis=1)

    if path_save:
        track_dir(path_save)
        dataframe.to_csv(path_save, index=False)

    return dataframe


def from_MT5(dataframe,
             path_save=None):
    dataframe.rename(
        columns={'time': 'DATE', 'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE',
                 'tick_volume': 'TICKVOL', 'real_volume': 'VOL', 'spread': 'SPREAD'}, inplace=True)
    dataframe['TIME'] = pd.to_datetime(dataframe['DATE'], unit='s', origin='unix').dt.strftime('%H:%M:%S')
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'], unit='s', origin='unix').dt.strftime('%Y-%m-%d')
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATE'] + ' ' + dataframe['TIME'])
    dataframe = dataframe.drop(['DATE', 'TIME', 'VOL'], axis=1)

    if path_save:
        track_dir(path_save)
        dataframe.to_csv(path_save, index=False)

    return dataframe
