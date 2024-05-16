import os
import shutil
import device
import config
import utils
import pandas as pd
import dragonfly_data
import dragonfly_model
from utils.files import clogger
from dateutil import parser


@clogger('Load source symbols')
def load_symbols(start_date=None, end_date=None, path=None, symbols=None):
    if path is None:
        path = config.PATH_SOURCE_DATA
    if start_date is None:
        start_date = config.SOURCE_LOAD_START_DATE
    if isinstance(start_date, str):
        start_date = parser.parse(start_date)
    if end_date is None:
        end_date = config.SOURCE_LOAD_END_DATE
    if isinstance(end_date, str):
        end_date = parser.parse(end_date)
    if symbols is None:
        symbols = config.SOURCE_SYMBOLS
    timeframes = config.SOURCE_TIMEFRAMES
    device.tools.cpu_map(dragonfly_data.loader.mt5_load_symbol,
                         [symbols, timeframes, path, end_date, None, start_date, config.PREFIX_BROKER_SYMBOL],
                         num_workers=int(os.cpu_count() * device.config.VAL_CORE))


@clogger('Clearing temporary folder')
def clear_temp():
    utils.files.clear_directory(directory=config.PATH_TEMP, del_directory=True)


@clogger('Search target trends')
def prepare_date_out():
    symbols = config.SOURCE_SYMBOLS
    path_save = f'{config.PATH_TEMP}/_segments'

    # поиск при помощи adx индикатора
    # device.tools.cpu_map(dragonfly_data.collector.create_data_out_adx,
    #                      [symbols, config.MAIN_COLUMN, path_save, config.PATH_SOURCE_DATA, ['pos', 'neg']],
    #                      num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    # поиск монотонностей со сглаживанием
    # device.tools.cpu_map(dragonfly_data.collector.create_data_out_linear,
    #                      [symbols, config.MAIN_COLUMN, path_save, config.PATH_SOURCE_DATA,
    #                       ['pos', 'neg'], [30 + n for n in range(30)], '30'],
    #                      num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    # поиск по локальным минимумам и максимумам
    # device.tools.cpu_map(dragonfly_data.collector.create_data_out_sprinter,
    #                      [symbols, config.MAIN_COLUMN, path_save, config.PATH_SOURCE_DATA,
    #                       ['pos', 'neg'], [60]],
    #                      num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    # регрессия
    # device.tools.cpu_map(dragonfly_data.collector.create_data_out_regress,
    #                      [symbols, config.MAIN_COLUMN, path_save, config.PATH_SOURCE_DATA],
    #                      num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    # регрессия (расстояние до локального экстремума)
    # device.tools.cpu_map(dragonfly_data.collector.create_data_out_regress_splinter,
    #                      [symbols, config.MAIN_COLUMN, path_save, config.PATH_SOURCE_DATA],
    #                      num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    # регрессия (окно экстремума)
    device.tools.cpu_map(dragonfly_data.collector.create_data_out_regress_window,
                         [symbols, config.MAIN_COLUMN, path_save, config.PATH_SOURCE_DATA],
                         num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    utils.files.concat_csv_files(directory_path=path_save, output_filename=config.PATH_DATE_OUT)

    df = pd.read_csv(config.PATH_DATE_OUT)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.sort_values(by=['DATETIME'])

    # df = dragonfly_data.utils.drop_dupl_class(df, min_periods=10)
    df.to_csv(config.PATH_DATE_OUT, index=False)


@clogger('Results date out')
def view_results_date_out():
    df = pd.read_csv(config.PATH_DATE_OUT)
    class_columns = [col for col in df.columns]
    statistics = {}
    for symbol_value in df['SYMBOL'].unique():
        subset_df = df[df['SYMBOL'] == symbol_value]
        row_data = {}
        for class_col in class_columns:
            if 'CLASS' in class_col:
                row_data[class_col] = subset_df[class_col].eq(1).sum()
            if 'REGRESS' in class_col:
                row_data[class_col] = len(subset_df[class_col])
                print(class_col, subset_df[class_col].max(), subset_df[class_col].min(), subset_df[class_col].mean(), subset_df[class_col].median())
                print(len(subset_df.loc[subset_df[class_col] > 1.0][class_col]), len(subset_df.loc[subset_df[class_col] < -1.0][class_col]))
        statistics[symbol_value] = row_data
    result_df = pd.DataFrame(statistics).T
    pd.set_option('display.max_columns', None)
    print(f'\n{result_df}\n')


@clogger('Create data train')
def prepare_data_train():
    utils.files.clear_directory(config.PATH_TRAIN_DATA, del_directory=True)

    # для классификации
    # device.tools.cpu_map(dragonfly_model.data.create_data_train_classification,
    #                      [config.SOURCE_SYMBOLS, [config.IN_TIMEFRAMES], [config.SELECTED_COLUMNS],
    #                       config.LEN_INPUT, config.PATH_SOURCE_DATA, config.PATH_TRAIN_DATA, config.PATH_DATE_OUT],
    #                      num_workers=int(os.cpu_count() * device.config.VAL_CORE))

    # для регрессии
    device.tools.cpu_map(dragonfly_model.data.create_data_train_regression,
                         [config.SOURCE_SYMBOLS, [config.IN_TIMEFRAMES], [config.SELECTED_COLUMNS],
                          config.LEN_INPUT, config.PATH_SOURCE_DATA, config.PATH_TRAIN_DATA, config.PATH_DATE_OUT],
                         num_workers=int(os.cpu_count() * device.config.VAL_CORE))


@clogger('Train CNN model')
def train_cnn(model_type='Kite', new=None, epochs=1000):
    utils.files.clear_directory(directory=config.PATH_TEMP_MODELS, del_directory=True)
    npz_train_files = utils.files.find_files(config.PATH_TRAIN_DATA, '', extension='.npz')
    device.tools.gpu_multifit(method=dragonfly_model.builder.fit_dragonfly,
                              combination_args=[config.PATH_TEMP_MODELS,
                                                npz_train_files,
                                                config.LEN_INPUT,
                                                len(config.IN_TIMEFRAMES),
                                                len(config.SELECTED_COLUMNS),
                                                config.SIZE_EYE,
                                                config.SIZE_WING,
                                                config.SIZE_TAIL,
                                                model_type,
                                                new,
                                                epochs,
                                                None],
                              devices=[0])


@clogger('Train Forest model')
def train_forest():
    utils.files.clear_directory(directory=config.PATH_TEMP_MODELS, del_directory=True)
    npz_train_files = utils.files.find_files(config.PATH_TRAIN_DATA, '', extension='.npz')
    for train_file in npz_train_files:
        dragonfly_model.builder.fit_forest(config.PATH_TEMP_MODELS, train_file)


@clogger('Copy models in production')
def copy_in_production():
    for root, dirs, files in os.walk(config.PATH_TEMP_MODELS):
        for file in files:
            if file == 'model.h5':
                file_path = os.path.join(root, file)
                new_file_name = os.path.basename(os.path.dirname(file_path)) + os.path.splitext(file)[1]
                des_path = os.path.join(config.PATH_PRODUCTION_MODELS, new_file_name)
                utils.files.track_dir(des_path)
                shutil.copy(file_path, des_path)
