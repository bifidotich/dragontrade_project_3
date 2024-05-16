########################################################################################################################
import os
import time
import utils
import joblib
import config
import device
import dragonfly_data
import dragonfly_model
import dragonfly_trade
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from datetime import timedelta
from utils.builder import datetime_now_timezone
########################################################################################################################

PATH_WORK = config.PATH_PRODUCTION
PATH_DATA = config.PATH_UPDATE_DATA
TRADE_SYMBOLS = config.TRADE_SYMBOLS
ACCURACY_OPEN = config.TRADE_PREDICT_OPENING_TRANSACTION
TZ = config.WORK_TIMEZONE

MODEL_CLASS = '3_30'
OPEN_ORDER_FILE = f'{PATH_WORK}/journal_orders.pkl'
PATH_PREDICT_LOG = f'{PATH_WORK}/history_predicts.txt'

########################################################################################################################


def load_model():
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # if gpu_devices:
    #     tf.config.experimental.set_visible_devices(gpu_devices[1], 'GPU')
    tf.config.experimental.set_visible_devices([], 'GPU')
    models = {}
    for syb in TRADE_SYMBOLS:
        try:
            model_file = f'{config.PATH_PRODUCTION_MODELS}/{syb}_CLASS_{MODEL_CLASS}.h5'
            models[syb] = dragonfly_model.cnn.load(model_file)
        except Exception as e:
            print(f"({datetime_now_timezone(TZ, delta=False)}) _load_model: {str(e)}")
    return models


def main():

    # обновляем модели
    models = load_model()
    time.sleep(60) if len(models) == 0 else time.sleep(0)

    for syb in TRADE_SYMBOLS:

        # проверяем что модель загружена
        if syb not in models:
            continue

        # обновляем котировки
        datetime_now = datetime_now_timezone(TZ, delta=True)
        start_date = datetime_now - timedelta(hours=(24 * 60))
        device.tools.cpu_map(dragonfly_data.loader.mt5_load_symbol,
                             [[syb], config.SOURCE_TIMEFRAMES, PATH_DATA, datetime_now, None,
                              start_date, config.PREFIX_BROKER_SYMBOL],
                             num_workers=int(os.cpu_count() * device.config.VAL_CORE))

        # проверяем актуальность котировок
        in_file = f"{PATH_DATA}/{syb}_M1.csv"
        datetime_data = dragonfly_data.utils.get_date_latest_data(df=pd.read_csv(in_file, sep=','),
                                                                  name_date_column='DATETIME',
                                                                  string=False)
        datetime_now = datetime_now_timezone(TZ, delta=False)
        if (datetime_now - datetime_data) > timedelta(seconds=60):
            print(f'({datetime_now}) Нет актуальных котировок {syb}, пропуск, (now data {datetime_data})')
            continue

        # делаем прогноз
        label, probability = None, None
        try:
            time_out_labels = np.array([np.datetime64(datetime_data)])

            dataframes = dragonfly_model.data.load_dataframes(loc_path_source=PATH_DATA,
                                                              loc_symbol=syb,
                                                              timeframes=config.IN_TIMEFRAMES)

            input_data = dragonfly_model.data.extend_segments_timeframes(dataframes=dataframes,
                                                                         time_out_labels=time_out_labels.copy(),
                                                                         sel_columns=config.SELECTED_COLUMNS,
                                                                         len_input=config.LEN_INPUT)

            out_predict = dragonfly_model.cnn.predict(model=models[syb], input_data=input_data)
            # print('out_predict', out_predict)
            out_predict = np.array(out_predict)
            labels = np.argmax(out_predict, axis=-1)
            label = labels[0]
            probability = out_predict[0][label]

            print(datetime_data, '|', (datetime_now - datetime_data), '|', syb, probability, label)

            import sys
            np.set_printoptions(threshold=sys.maxsize)
            print(input_data)

        except Exception as e:
            print(f"({datetime_now_timezone(TZ, delta=False)}) Внутренняя ошибка прогноза {syb}: {str(e)}")
            continue

        # переводим метки
        if label == 0:
            direction = 'down'
            stop_loss = None
            take_profit = None
        elif label == 1:
            direction = 'up'
            stop_loss = None
            take_profit = None
        else:
            continue

        # загружаем рабочий журнал
        journal = dragonfly_trade.mt5.load_journal(path_file=OPEN_ORDER_FILE)
        journal.max_open_orders = config.TRADE_MAX_OPEN_ORDERS
        journal.check_orders()

        # закрываем противоположные позиции
        if probability > 0.9:
            journal.close_orders(symbol=syb, direction=dragonfly_trade.mt5.inverse_direction(direction=direction, lower=True))

        # открываем новые позиции
        if probability >= ACCURACY_OPEN:
            # создаем заявку на ордер
            trade_volume = config.TRADE_VOLUME
            minute_life = 180
            datetime_kill = datetime_now + timedelta(minutes=minute_life)
            journal.create_order(symbol=syb,
                                 direction=direction,
                                 volume=trade_volume,
                                 stop_loss=stop_loss,
                                 take_profit=take_profit,
                                 datetime_kill=datetime_kill)

        joblib.dump(journal, OPEN_ORDER_FILE)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"({datetime_now_timezone(TZ, delta=False)}) [main_worker] Произошла ошибка: {str(e)}")
            time.sleep(10)
            continue
