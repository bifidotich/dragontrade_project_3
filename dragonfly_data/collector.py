
import random
import warnings
import numpy as np
import pandas as pd
import utils as utl
import dragonfly_data
import dragonfly_trade
from ta.trend import adx, adx_neg, adx_pos


def create_data_out_adx(symbol,
                        main_column,
                        path_out_directory,
                        path_source,
                        direction,  # direction 'pos', 'neg'
                        min_max_duration=(60, 100),
                        threshold=20):
    mmd = list(min_max_duration)
    class_name = str(min_max_duration[-1])
    factor_spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=symbol) * 6

    df_results = pd.DataFrame(columns=['SYMBOL', 'DATETIME'])

    df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df['MEAN_MAIN'] = df[main_column].rolling(window=int(mmd[0] // 10), center=True).mean()

    adx_name = f'adx_{direction}'.upper()
    if adx_name == 'ADX_POS':
        direction_num = 1
        direction_name = 'LONG'
    elif adx_name == 'ADX_NEG':
        direction_num = 0
        direction_name = 'SHORT'
    else:
        raise ValueError(f'adx_name "{direction}" not supported, only "pos" or "neg"')

    row_dst, value_start, dur_trend, idx_start, date_start = None, None, 0, 0, None
    for idx, row in df.iterrows():
        if row[adx_name] > threshold:
            if dur_trend == 0:
                row_dst = row
                value_start = row[main_column]
                date_start = row['DATETIME']
            dur_trend += 1
        elif row['ADX'] < threshold:
            if row_dst is not None:
                if abs(value_start - row[main_column]) > factor_spread and mmd[0] < dur_trend < mmd[1]:
                    new_row = {'DATETIME': date_start, 'SYMBOL': symbol,
                               f'CLASS_5_{class_name}_{direction_name}_OPEN': 1.0}
                    df_results = df_results._append(new_row, ignore_index=True)

                    new_row_close = {'DATETIME': row['DATETIME'], 'SYMBOL': symbol,
                                     f'CLASS_5_{class_name}_{direction_name}_CLOSE': 1.0}
                    df_results = df_results._append(new_row_close, ignore_index=True)

            elif random.randint(0, 20) == 0:

                new_row = {'DATETIME': row['DATETIME'], 'SYMBOL': symbol,
                           f'CLASS_5_{class_name}_{direction_name}_NAN': 1.0}
                df_results = df_results._append(new_row, ignore_index=True)

            dur_trend = 0
            row_dst, date_start = None, None

    df_results = df_results.fillna(0.0)
    path_file = f'{path_out_directory}/{symbol}_{direction_num}.csv'
    utl.files.track_dir(path_file)
    df_results.to_csv(path_file, index=False)


def create_data_out_linear(symbol,
                           main_column,
                           path_out_directory,
                           path_source,
                           direction,
                           len_segment,
                           class_name='',
                           factor_spread=8):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=symbol)
    threshold_change = factor_spread * spread
    num_next_columns = len_segment + int(len_segment / 2)
    columns_next = [f'NEXT_VALUE_{idx}' for idx in range(1, num_next_columns)]

    error_spread = 7e-1 * spread
    window_size = int(len_segment // 8)
    min_period = int(len_segment // 12)

    if direction == 'pos':
        direction_num = 1
        direction_name = 'LONG'
    elif direction == 'neg':
        direction_num = 0
        direction_name = 'SHORT'
    else:
        raise ValueError(f'adx_name "{direction}" not supported, only "pos" or "neg"')

    df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=False)

    df['VALUE'] = df[main_column].rolling(window=window_size).mean()
    df['NEXT_DATETIME'] = df['DATETIME'].shift(-len_segment)
    for idx in range(1, num_next_columns):
        df[f'NEXT_VALUE_{idx}'] = df['VALUE'].shift(-idx)

    df = df[df['NEXT_DATETIME'] - df['DATETIME'] == pd.Timedelta(minutes=len_segment)]
    if direction_num == 0:
        df = df[df[f'NEXT_VALUE_{len_segment}'] - df['VALUE'] < -threshold_change]
        df = df[df[f'NEXT_VALUE_{len_segment}'] == df[columns_next].min(axis=1)]
        for idx in range(1, len_segment):
            df = df[df[f'NEXT_VALUE_{idx}'] > df[f'NEXT_VALUE_{idx + 1}'] - error_spread]
    elif direction_num == 1:
        df = df[df[f'NEXT_VALUE_{len_segment}'] - df['VALUE'] > threshold_change]
        df = df[df[f'NEXT_VALUE_{len_segment}'] == df[columns_next].max(axis=1)]
        for idx in range(1, len_segment):
            df = df[df[f'NEXT_VALUE_{idx}'] < df[f'NEXT_VALUE_{idx + 1}'] + error_spread]
    df = df[(df['DATETIME'] + pd.Timedelta(minutes=min_period) < df['DATETIME'].shift(-1))]

    df_results_open = pd.DataFrame({'DATETIME': df['DATETIME'],
                                    'SYMBOL': [symbol] * len(df),
                                    f'CLASS_3_{class_name}_{direction_name}_OPEN': 1.0})
    df_results_close = pd.DataFrame({'DATETIME': df['NEXT_DATETIME'],
                                     'SYMBOL': [symbol] * len(df),
                                     f'CLASS_3_{class_name}_{direction_name}_CLOSE': 1.0})
    df_results = pd.concat([df_results_open, df_results_close], ignore_index=True)

    df_results = df_results.fillna(0.0)
    df_results['DATETIME'] = pd.to_datetime(df_results['DATETIME'])
    df_results.sort_values(by='DATETIME', inplace=True)

    path_file = f'{path_out_directory}/{symbol}_{direction_num}_{len_segment}.csv'
    utl.files.track_dir(path_file)
    df_results.to_csv(path_file, index=False)


def create_data_out_sprinter(symbol,
                             main_column,
                             path_out_directory,
                             path_source,
                             direction,
                             duration,
                             factor_spread=6):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=symbol)
    threshold_change = factor_spread * spread
    len_segment = int(duration)
    min_period = int(len_segment // 2)

    if direction == 'pos':
        direction_num = 1
        direction_name = 'LONG'
    elif direction == 'neg':
        direction_num = 0
        direction_name = 'SHORT'
    else:
        raise ValueError(f'adx_name "{direction}" not supported, only "pos" or "neg"')

    df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=False)

    df['VALUE'] = df[main_column] + np.random.uniform(0, 1 / pow(10, 8), size=len(df))

    df['VALUE_MAX'] = df['VALUE'].shift(-len_segment).rolling(window=len_segment, center=False).max()
    df['VALUE_MIN'] = df['VALUE'].shift(-len_segment).rolling(window=len_segment, center=False).min()

    df['MAX_INDEX'] = df['VALUE'].shift(-len_segment).rolling(len_segment, center=False).apply(np.argmax,
                                                                                               raw=True) + np.arange(
        len(df))
    df['MIN_INDEX'] = df['VALUE'].shift(-len_segment).rolling(len_segment, center=False).apply(np.argmin,
                                                                                               raw=True) + np.arange(
        len(df))
    df = df.fillna(0)
    df['MAX_INDEX'] = df['MAX_INDEX'].astype(int)
    df['MIN_INDEX'] = df['MIN_INDEX'].astype(int)
    df['MAX_INDEX'] = df['MAX_INDEX'].clip(upper=len(df) - 1)
    df['MIN_INDEX'] = df['MIN_INDEX'].clip(upper=len(df) - 1)
    df['DATETIME_MAX'] = df.loc[df['MAX_INDEX'], 'DATETIME'].values
    df['DATETIME_MIN'] = df.loc[df['MIN_INDEX'], 'DATETIME'].values

    df = df[df[f'VALUE_MAX'] - df['VALUE_MIN'] > threshold_change]
    if direction_num == 0:
        df = df[df['DATETIME_MIN'] - df['DATETIME_MAX'] >= pd.Timedelta(minutes=min_period)]
        df['DATETIME_OPEN'] = df['DATETIME_MAX']
        df['DATETIME_CLOSE'] = df['DATETIME_MIN']
    elif direction_num == 1:
        df = df[df['DATETIME_MAX'] - df['DATETIME_MIN'] >= pd.Timedelta(minutes=min_period)]
        df['DATETIME_OPEN'] = df['DATETIME_MIN']
        df['DATETIME_CLOSE'] = df['DATETIME_MAX']
    df = df[(df['DATETIME'] + pd.Timedelta(minutes=int(len_segment // 10)) <= df['DATETIME'].shift(-1))]

    df_results_open = pd.DataFrame({'DATETIME': df['DATETIME_OPEN'] + pd.Timedelta(minutes=5),
                                    'SYMBOL': [symbol] * len(df),
                                    f'CLASS_4_{len_segment}_{direction_name}_OPEN': 1.0})
    df_results_close = pd.DataFrame({'DATETIME': df['DATETIME_CLOSE'] + pd.Timedelta(minutes=5),
                                     'SYMBOL': [symbol] * len(df),
                                     f'CLASS_4_{len_segment}_{direction_name}_CLOSE': 1.0})
    df_results = pd.concat([df_results_open, df_results_close], ignore_index=True)

    df_results = df_results.fillna(0.0)
    df_results['DATETIME'] = pd.to_datetime(df_results['DATETIME'])
    df_results.sort_values(by='DATETIME', inplace=True)

    df_results = dragonfly_data.utils.drop_dupl_class(df_results, min_periods=min_period)

    path_file = f'{path_out_directory}/{symbol}_{direction_num}_{len_segment}.csv'
    utl.files.track_dir(path_file)
    df_results.to_csv(path_file, index=False)


def create_data_out_regress(symbol,
                            main_column,
                            path_out_directory,
                            path_source,
                            target_duration=60,
                            target_factor=8,
                            period=10):
    spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=symbol)

    df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df, window_smooth=0)
    df['MEAN_MAIN'] = df[main_column].rolling(window=int(target_duration // 10), center=True).mean()
    df['DIFF_MAIN'] = df['MEAN_MAIN'].shift(-target_duration) - df['MEAN_MAIN']
    df = df.iloc[::period]
    # df['REGRESS_1'] = pow(df['DIFF_MAIN'] / spread / target_factor, 9)
    # df['REGRESS_1'] = df['DIFF_MAIN'] / (spread * target_factor)
    df['REGRESS_1'] = pow((df['DIFF_MAIN'] / spread) / target_factor, 3)
    df['SYMBOL'] = symbol

    df_results = df[['DATETIME', 'SYMBOL', 'REGRESS_1']]
    print(len(df), (df['REGRESS_1'] > 1).sum(), (df['REGRESS_1'] < -1).sum())
    path_file = f'{path_out_directory}/{symbol}_REGRESS_{target_duration}.csv'
    utl.files.track_dir(path_file)
    df_results.to_csv(path_file, index=False)


def create_data_out_regress_splinter(symbol,
                                     main_column,
                                     path_out_directory,
                                     path_source,
                                     window_size=90,
                                     target_factor=8,
                                     period=10):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=symbol)

    df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=False)
    df['value'] = df[main_column].rolling(window=int(20), center=True).mean()
    df['shift_value'] = df['value'].shift(-window_size)

    df['max'] = df['shift_value'].rolling(window=window_size).max()
    df['index_max'] = df['shift_value'].rolling(window=window_size).apply(np.argmax, raw=True) + np.arange(len(df))

    df['min'] = df['shift_value'].rolling(window=window_size).min()
    df['index_min'] = df['shift_value'].rolling(window=window_size).apply(np.argmin, raw=True) + np.arange(len(df))

    df = df.dropna(subset=['max', 'min'])
    df = df.iloc[::period]
    df['index_max'] = df['index_max'].astype(int)
    df['index_min'] = df['index_min'].astype(int)

    df.loc[df['index_max'] <= period + df.index, 'index_min'] = 0
    df.loc[df['index_min'] <= period + df.index, 'index_max'] = 0

    df_pos = df[(df['index_max'] < df['index_min'])]
    df_neg = df[(df['index_max'] > df['index_min'])]

    df_pos['res'] = df_pos['max'] - df_pos['value']
    df_neg['res'] = df_neg['min'] - df_neg['value']
    df = pd.concat([df_pos, df_neg])

    df['SYMBOL'] = symbol
    # df['REGRESS_2'] = df['res'] / spread
    df['REGRESS_2'] = pow((df['res'] / spread) / target_factor, 3)
    df_results = df[['DATETIME', 'SYMBOL', 'REGRESS_2']]

    print(len(df), (df['REGRESS_2'] > 1).sum(), (df['REGRESS_2'] < -1).sum())
    path_file = f'{path_out_directory}/{symbol}_REGRESS_{window_size}.csv'
    utl.files.track_dir(path_file)
    df_results.to_csv(path_file, index=False)


def create_data_out_regress_window(symbol,
                                   main_column,
                                   path_out_directory,
                                   path_source,
                                   window_size=60,
                                   window_smooth=10,
                                   target_factor=6):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=symbol)

    df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=False)
    df['value'] = df[main_column].rolling(window=window_smooth, center=True).mean()
    df['shift_value'] = df['value'].shift(-window_size)
    df['value_max'] = df['shift_value'].rolling(window=window_size, center=False).max()
    df['value_min'] = df['shift_value'].rolling(window=window_size, center=False).min()

    df['adx'] = adx(df['HIGH'], df['LOW'], df['CLOSE'], fillna=True)
    df = df.iloc[::10]
    df = df[(df['adx'] > 20)]

    df['diff_pos'] = df['value_max'] - df['value']
    df['diff_neg'] = df['value_min'] - df['value']

    # df['res'] = df['shift_value'] - df['value']
    df['res'] = df[['diff_pos', 'diff_neg']].abs().max(axis=1) * np.sign(df[['diff_pos', 'diff_neg']].to_numpy()).max(axis=1)

    df['SYMBOL'] = symbol
    # df['REGRESS_2'] = df['res'] / spread
    df['REGRESS_2'] = pow((df['res'] / spread) / target_factor, 3)
    df_results = df[['DATETIME', 'SYMBOL', 'REGRESS_2']]

    print(len(df), (df['REGRESS_2'] > 1).sum(), (df['REGRESS_2'] < -1).sum())
    path_file = f'{path_out_directory}/{symbol}_REGRESS_{window_size}.csv'
    utl.files.track_dir(path_file)
    df_results.to_csv(path_file, index=False)
