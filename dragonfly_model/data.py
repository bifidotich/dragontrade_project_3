import os
import utils
import numpy as np
import pandas as pd
import dragonfly_data
import dragonfly_model
from dragonfly_data.builder import MidPositionScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.utils import shuffle, class_weight
from utils.files import find_files, clogger


def load_dataframes(loc_path_source, loc_symbol, timeframes):
    loc_dataframes = [pd.read_csv(f'{loc_path_source}/{loc_symbol}_{it_tf}.csv') for it_tf in timeframes]
    loc_dataframes = [dragonfly_data.convertor.to_dragonfly_format(dataframe=item_df) for item_df in loc_dataframes]
    loc_dataframes = [dragonfly_data.builder.preparer_datetime(dataframe=item_df) for item_df in loc_dataframes]
    return loc_dataframes


def extend_segments_timeframes(dataframes, time_out_labels, sel_columns, len_input):
    def _find_indices_before_values(dataframe, datetime_index):
        idx = pd.to_datetime(dataframe.index)
        datetime_indices = pd.to_datetime(datetime_index)
        result_indices = []
        for dt_index in datetime_indices:
            if dt_index not in idx:
                dt_index = idx[dt_index > idx][-1]
            result_indices.append(dataframe.index.get_loc(dt_index) - 1)
        return np.array(result_indices)

    segments_timeframes = []
    for df_idx, df_timeframe in enumerate(dataframes):

        # df_timeframe[sel_columns] = MidPositionScaler(position=0.2).fit_transform(df_timeframe[sel_columns], cut=True)
        # df_timeframe[sel_columns] = RobustScaler().fit_transform(df_timeframe[sel_columns])
        # df_timeframe[sel_columns] = MinMaxScaler().fit_transform(df_timeframe[sel_columns])
        # df_timeframe[sel_columns] = StandardScaler().fit_transform(df_timeframe[sel_columns])

        end_indices = _find_indices_before_values(df_timeframe, time_out_labels.copy())
        start_indices = np.maximum(0, end_indices - len_input + 1)
        rows_slices = [slice(start, end + 1) for start, end in zip(start_indices, end_indices)]
        cols_slices = [df_timeframe.columns.get_loc(col) for col in sel_columns]
        subarrays = [df_timeframe.iloc[rows, cols_slices].values for rows in rows_slices]
        subarrays = np.array(subarrays).transpose((2, 0, 1))
        segments_timeframes.append(np.array(subarrays))

    segments_timeframes = np.array(segments_timeframes).transpose((2, 0, 3, 1))

    # min_value = np.min(segments_timeframes, axis=2, keepdims=True)
    # max_value = np.max(segments_timeframes, axis=2, keepdims=True)
    # segments_timeframes = (segments_timeframes - min_value) / (max_value - min_value)

    return segments_timeframes


def create_data_train_classification(symbol,
                                     timeframes,
                                     selected_columns,
                                     len_input,
                                     path_prod_tick,
                                     path_out_directory,
                                     path_date_out,
                                     class_filter=''):
    df = pd.read_csv(f'{path_prod_tick}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=True)
    size_obr = dragonfly_data.utils.timeframe_string_to_minutes(timeframes[-1])
    obr_time = pd.to_datetime(df.index.min() + pd.Timedelta(minutes=int(size_obr * len_input * 2)))

    # list_sample_class = ['_SHORT', '_LONG']
    list_sample_class = ['_OPEN', '_CLOSE', '_NAN']
    # list_sample_class = ['_OPEN', '_CLOSE']

    df = pd.read_csv(path_date_out)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.loc[df['SYMBOL'] == symbol]
    df = df.loc[df['DATETIME'] > obr_time]

    full_class_list = [_ for _ in df.columns if 'CLASS_' in _ and class_filter in _]
    for replacement in list_sample_class:
        full_class_list = [item.replace(replacement, '') for item in full_class_list]
    unique_class_list = list({s for s in full_class_list})

    for iucl in unique_class_list:

        if 'LONG' not in iucl:  # TODO
            continue

        class_list = [f'{iucl}{s}' for s in list_sample_class]
        print(class_list)
        filtered_df = df.copy()[df[class_list].ne(0).any(axis=1)]
        time_out_labels = filtered_df['DATETIME'].values.astype(np.datetime64)
        out_labels = np.array(filtered_df[class_list].to_numpy())
        time_out_labels, out_labels = utils.builder.zip_sorted(time_out_labels, out_labels)

        if len(time_out_labels) > 0:
            dataframes = load_dataframes(loc_path_source=path_prod_tick,
                                         loc_symbol=symbol,
                                         timeframes=timeframes)

            segments_timeframes = extend_segments_timeframes(dataframes=dataframes,
                                                             time_out_labels=time_out_labels,
                                                             sel_columns=selected_columns,
                                                             len_input=len_input)

            indices = np.random.permutation(len(out_labels))
            in_data = segments_timeframes[indices]
            out_data = out_labels[indices]
            labels = out_labels[indices]

            path_out_file = f'{path_out_directory}/{symbol}_{iucl}.npz'
            utils.files.track_dir(path_out_file)
            np.savez(path_out_file, in_data=in_data, out_data=out_data, labels=labels)


def create_data_train_regression(symbol,
                                 timeframes,
                                 selected_columns,
                                 len_input,
                                 path_prod_tick,
                                 path_out_directory,
                                 path_date_out):
    df = pd.read_csv(f'{path_prod_tick}/{symbol}_M1.csv')
    df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
    df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=True)
    size_obr = dragonfly_data.utils.timeframe_string_to_minutes(timeframes[-1])
    obr_time = pd.to_datetime(df.index.min() + pd.Timedelta(minutes=int(size_obr * len_input * 2)))

    df = pd.read_csv(path_date_out)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.loc[df['SYMBOL'] == symbol]
    df = df.loc[df['DATETIME'] > obr_time]

    full_class_list = [_ for _ in df.columns if 'REGRESS_' in _]
    unique_class_list = list({s for s in full_class_list})

    for iucl in unique_class_list:

        class_list = [iucl]
        print(class_list)
        filtered_df = df.copy().dropna(subset=[iucl])
        time_out_labels = filtered_df['DATETIME'].values.astype(np.datetime64)
        out_labels = np.array(filtered_df[class_list].to_numpy())
        time_out_labels, out_labels = utils.builder.zip_sorted(time_out_labels, out_labels)

        if len(time_out_labels) > 0:
            dataframes = load_dataframes(loc_path_source=path_prod_tick,
                                         loc_symbol=symbol,
                                         timeframes=timeframes)

            segments_timeframes = extend_segments_timeframes(dataframes=dataframes,
                                                             time_out_labels=time_out_labels,
                                                             sel_columns=selected_columns,
                                                             len_input=len_input)

            indices = np.random.permutation(len(out_labels))
            in_data = segments_timeframes[indices]
            out_data = out_labels[indices]
            labels = out_labels[indices]

            path_out_file = f'{path_out_directory}/{symbol}_{iucl}.npz'
            utils.files.track_dir(path_out_file)
            np.savez(path_out_file, in_data=in_data, out_data=out_data, labels=labels)
