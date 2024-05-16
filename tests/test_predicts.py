import os
import random
import time
import config
import research
import utils
import factory
import datetime
import dateutil.parser
import numpy as np
import pandas as pd
import tensorflow as tf
import dragonfly_model
import dragonfly_trade

########################################################################################################################

syb = 'EURUSD'
model_class = 'REGRESS_2'
size_period = 90
datetime_end = dateutil.parser.parse('2024-02-26 00:00:00')

########################################################################################################################

if __name__ == '__main__':
    utils.files.set_project_directory()
    utils.files.mother_iam_coder()

    # factory.load_symbols(start_date='2023-06-01 00:00:00', end_date='2024-04-01 00:00:00', path=config.PATH_UPDATE_DATA)

    spread = dragonfly_trade.mt5.mt5_mean_spread(symbol=syb)
    df_results = pd.DataFrame(columns=['datetime', 'label', 'probability', 'value'])
    dataframes = dragonfly_model.data.load_dataframes(loc_path_source=config.PATH_UPDATE_DATA,
                                                      loc_symbol=syb,
                                                      timeframes=config.IN_TIMEFRAMES)

########################################################################################################################

    time_out_labels = []
    print('datetime_start', datetime_end)
    while True:
        datetime_end = datetime_end + datetime.timedelta(minutes=1)
        time_out_labels.append(np.datetime64(datetime_end))
        if dataframes[0].index.values[-1] < np.datetime64(datetime_end):
            break
    print('datetime_end', datetime_end)
    time_out_labels = np.array(time_out_labels)
    # time_out_labels = np.array([dateutil.parser.parse('2024-02-16 00:39:00')])

########################################################################################################################

    input_data = dragonfly_model.data.extend_segments_timeframes(dataframes=dataframes.copy(),
                                                                 time_out_labels=time_out_labels.copy(),
                                                                 sel_columns=config.SELECTED_COLUMNS,
                                                                 len_input=config.LEN_INPUT)

    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # if gpu_devices:
    #     tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')
    #
    # model = dragonfly_model.cnn.load(f'{config.PATH_TEMP_MODELS}/{syb}_{model_class}/model.h5')
    # out_predicts = dragonfly_model.cnn.predict(model=model, input_data=input_data.copy())
    # label = out_predicts.reshape(-1)

    input_data, out_data = dragonfly_model.utils.flatten_data(input_data)

    directory_model = f'{config.PATH_TEMP_MODELS}/{syb}_{model_class}_FOREST'
    out_predicts = dragonfly_model.builder.predict_forest(directory_model=directory_model, data=input_data)
    label = out_predicts

########################################################################################################################

    # start, step = 0.5, 0.01
    # source_tpot = [start + i * step for i in range(int((1.0 - start) // step) + 1)]
    # source_tpot += [0.999, 1.0]
    print_list = []
    source_tpot = [0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

    list_description, list_count_results, list_accuracy_results, list_profit, list_time_num = [], [], [], [], []
    for tpot in source_tpot:

        # psos = tpot
        # tpot = 0.99

        df_check = dataframes[0]
        all_predict, true_predict, profit, dangers, cur_dur = 0, 0, 0, 0, 0
        cur_time, open_time, close_time, open_val = None, None, None, None

        print('=' * 100)
        print('labels', label.shape, label.max(), label.min(), label.mean())
        for idx, iter_val in enumerate(label):

            cur_dur += 1

            cur_time = time_out_labels[idx]
            if cur_time > df_check.index.values[-1]:
                break
            if cur_time not in df_check.index:
                continue

            if open_time is not None:

                val_current = df_check.loc[open_time, 'OPEN']
                val_future = df_check.loc[cur_time, 'CLOSE']

                danger = False
                if open_time is not None:

                    if open_val > 0 and 1 > iter_val and cur_dur > 10:
                        danger = True
                    elif open_val < 0 and 1 < iter_val and cur_dur > 10:
                        danger = True

                if danger: dangers += 1

                if cur_time >= close_time or danger:

                    is_true = False
                    if open_val < 0 and val_current > val_future + spread:
                        is_true = True
                    elif open_val > 0 and val_future > val_current + spread:
                        is_true = True

                    if is_true:
                        true_predict += 1
                        profit = profit + (abs(val_future - val_current) - spread)
                        list_time_num.append(open_time.astype(datetime.datetime).hour)
                    else:
                        profit = profit - (abs(val_future - val_current) + spread)
                    all_predict += 1

                    print(open_val, open_time, cur_time, is_true)
                    open_time, close_time, open_probability, open_label = None, None, None, None

                # if abs(iter_val) > tpot:
                #     cur_dur = 0

            if open_time is None and abs(iter_val) > tpot:
                open_time = time_out_labels[idx]
                open_val = iter_val
                close_time = open_time + np.timedelta64(size_period, 'm')
                cur_dur = 0

        round_tpot = round(tpot, 3)
        round_ta = round(true_predict / all_predict if all_predict > 0 else 0, 3)
        round_ps = round(profit / spread, 3)

        print_list.append(f'prob {round_tpot}, dangers {dangers}, accuracy {all_predict} {true_predict} {round_ta}, profit {profit}, {round_ps}')
        print(print_list[-1])

        list_count_results.append(all_predict)
        list_accuracy_results.append(round_ta)
        list_description.append(tpot)
        list_profit.append(round_ps)

    print('=' * 100)
    for _ in print_list: print(_)
    list_count_results = [x / max(list_count_results + [1]) for x in list_count_results]
    list_profit = [x / max(list_profit + [1]) for x in list_profit]
    list_profit = [x if x > 0 else 0.0 for x in list_profit]
    list_accuracy_results = [list_accuracy_results[idx] if list_count_results[idx] > 0 else 0.0 for idx in
                             range(len(list_accuracy_results))]
    research.SimpleGraph([list_count_results, list_accuracy_results, list_profit], x_line=list_description)
    # research.DistributionGraph(list_time_num)
