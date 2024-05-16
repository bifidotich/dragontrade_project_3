import os
import joblib
import datetime
import numpy as np
import utils as utl
import dragonfly_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def fit_dragonfly(directory_models,
                  path_train_file,
                  length_input,
                  count_timeframes,
                  count_features,
                  size_eye,
                  size_wing,
                  size_tail,
                  model_type='Wyvern',
                  new=False,
                  epochs=1000,
                  idx_device=0):
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices and idx_device is not None:
        dev = gpu_devices[idx_device] if idx_device is not None else []
        tf.config.experimental.set_visible_devices(dev, 'GPU')

    file_name_without_extension = os.path.splitext(os.path.basename(path_train_file))[0]
    directory_model = f'{directory_models}/{file_name_without_extension}'

    path_model = f"{directory_model}/model.h5"
    path_model_checkpoint = f"{directory_model}/_checkpoints"
    path_model_checkpoint_file = path_model_checkpoint + "/model_epoch_{epoch:02d}.h5"
    path_loger = f"{directory_model}/_logs/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    path_diagram_model = f'{directory_model}/model.png'
    utl.files.track_dir(path_loger)
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Fit model: {model_type} start')

    data = np.load(path_train_file)
    count_out = len(data['out_data'][0])

    if not os.path.exists(path_model) or new:

        if model_type == 'Wyvern':
            model = dragonfly_model.cnn.Wyvern(count_timeframes=count_timeframes,
                                               length_input=length_input,
                                               count_features=count_features,
                                               count_out=count_out,
                                               size_wing=size_wing,
                                               size_tail=size_tail)
        elif model_type == 'Chimera':
            model = dragonfly_model.cnn.Chimera(count_timeframes=count_timeframes,
                                                length_input=length_input,
                                                count_features=count_features,
                                                count_out=count_out,
                                                size_wing=size_wing,
                                                size_tail=size_tail)
        elif model_type == 'Pigeon':
            model = dragonfly_model.cnn.Pigeon(count_timeframes=count_timeframes,
                                               length_input=length_input,
                                               count_features=count_features,
                                               count_out=count_out,
                                               size_tail=size_tail)
        elif model_type == 'Manticore':
            model = dragonfly_model.cnn.Manticore(count_timeframes=count_timeframes,
                                                  length_input=length_input,
                                                  count_features=count_features,
                                                  count_out=count_out,
                                                  size_wing=size_wing,
                                                  size_tail=size_tail)
        elif model_type == 'Eagle':
            model = dragonfly_model.cnn.Eagle(count_timeframes=count_timeframes,
                                              length_input=length_input,
                                              count_features=count_features,
                                              count_out=count_out,
                                              size_wing=size_wing,
                                              size_tail=size_tail,
                                              size_eye=size_eye)
        elif model_type == 'Hawk':
            model = dragonfly_model.cnn.Hawk(count_timeframes=count_timeframes,
                                             length_input=length_input,
                                             count_features=count_features,
                                             count_out=count_out,
                                             size_wing=size_wing,
                                             size_tail=size_tail,
                                             size_eye=size_eye)
        elif model_type == 'Kite':
            model = dragonfly_model.cnn.Kite(count_timeframes=count_timeframes,
                                             length_input=length_input,
                                             count_features=count_features,
                                             count_out=count_out,
                                             size_wing=size_wing,
                                             size_tail=size_tail)
        elif model_type == 'Peacock':
            model = dragonfly_model.cnn.Peacock(count_timeframes=count_timeframes,
                                                length_input=length_input,
                                                count_features=count_features,
                                                count_out=count_out,
                                                size_wing=size_wing,
                                                size_tail=size_tail,
                                                size_eye=size_eye)
        elif model_type == 'Lera':
            model = dragonfly_model.cnn.Lera(count_timeframes=count_timeframes,
                                             length_input=length_input,
                                             count_features=count_features,
                                             count_out=count_out,
                                             size_wing=size_wing,
                                             size_tail=size_tail,)
        else:
            raise TypeError(f'model_type {model_type} not supported')

    else:
        model = dragonfly_model.cnn.load(path_model=path_model)

    if path_diagram_model is not None:
        dragonfly_model.cnn.draw(model, path=path_diagram_model)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=path_model_checkpoint_file,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True
    )

    validation_back_callback = dragonfly_model.cnn.ValidationBack(
        monitor='val_loss',
        increase=False,
        min_factor=0.03,
        verbose=1)

    csv_logger_callback = CSVLogger(
        filename=path_loger,
        separator=',',
        append=True)

    train_data, val_data, train_labels, val_labels = train_test_split(
        data['in_data'], data['out_data'], test_size=0.1, random_state=42)

    if count_out > 1:
        y = np.stack(data['out_data']).argmax(axis=1)
        class_weight_list = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {k: v for k, v in zip(np.unique(y), class_weight_list)}
    else:
        class_weight_dict = None

    print(f'Train model: {model_type} start')
    print(f'Train file: {path_train_file} start')
    print(f'Class weight: {class_weight_dict}')
    print(f'Distribution data: {np.unique([np.argmax(k) for k in data["out_data"]], return_counts=True)}')
    model.fit(train_data, train_labels,
              epochs=epochs,
              callbacks=[model_checkpoint_callback,
                         csv_logger_callback,
                         validation_back_callback,
                         early_stopping_callback],
              validation_data=(val_data, val_labels),
              class_weight=class_weight_dict,
              verbose=2)

    model.evaluate(val_data, val_labels, verbose=2)

    dragonfly_model.utils.export_model_file(source_dir=path_model_checkpoint, new_file_name=path_model)
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Fit model: {model_type} .....completed')
    del model, data, train_data, val_data, train_labels, val_labels


def predict_dragonfly(path_model, path_data, path_results, idx_device=0):
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        dev = gpu_devices[idx_device] if idx_device is not None else []
        tf.config.experimental.set_visible_devices(dev, 'GPU')

    data = np.load(path_data)
    model = dragonfly_model.cnn.load(path_model)
    predictions = model.predict(data['in_data'], verbose=2)
    np.save(path_results, predictions)
    del model
    return predictions


def fit_forest(directory_models, path_train_file):

    file_name_without_extension = os.path.splitext(os.path.basename(path_train_file))[0]
    directory_model = f'{directory_models}/{file_name_without_extension}_FOREST'
    path_model = f"{directory_model}/model.pkl"
    utl.files.track_dir(path_model)
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Fit forest: start')

    data = np.load(path_train_file)
    data_in, data_out = dragonfly_model.utils.flatten_data(data['in_data'], data['out_data'])
    train_data, val_data, train_labels, val_labels = train_test_split(data_in, data_out, test_size=0.1, random_state=42)

    print(f'Train file: {path_train_file} start')
    print('train_data', train_data.shape, 'train_out', train_labels.shape)
    print(f'Distribution data: {np.unique([np.argmax(k) for k in data["out_data"]], return_counts=True)}')

    model = RandomForestRegressor(n_estimators=100,
                                  random_state=42,
                                  n_jobs=20,
                                  # max_features=0.5,
                                  # min_samples_leaf=100,
                                  )
    model.fit(train_data, train_labels)

    y_pred = model.predict(val_data)
    mse = mean_squared_error(val_labels, y_pred)
    print(f"Среднеквадратичная ошибка: {mse:.3f}")

    joblib.dump(model, path_model)
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Fit forest: .....completed')
    del model, data, train_data, val_data, train_labels, val_labels


def predict_forest(directory_model, data):
    path_model = f"{directory_model}/model.pkl"
    model = joblib.load(path_model)
    prediction = model.predict(data)
    del model
    return prediction
