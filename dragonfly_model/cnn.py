import tensorflow as tf
from PIL import Image
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, Reshape, Concatenate
from keras.layers import Bidirectional, GRU, Attention
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D
from keras.layers import LSTM, Attention, Embedding, TimeDistributed
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop, SGD
from keras.regularizers import l1, l2
from keras.metrics import Precision
from keras.callbacks import Callback
from sklearn.ensemble import RandomForestRegressor


class ValidationBack(Callback):
    def __init__(self, monitor='val_loss', increase=False, min_factor=0.1, verbose=1):
        super(ValidationBack, self).__init__()
        self.monitor = monitor
        self.increase = increase
        self.min_factor = min_factor
        self.verbose = verbose
        self.best_weights = None
        self.best_metric = 0

    def on_epoch_end(self, epoch, logs=None):
        cm = logs.get(self.monitor)
        if cm is None:
            raise ValueError(f"Validation metric '{self.monitor}' not found in logs.")

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()
            self.best_metric = cm
        elif cm > 0.01:
            bm = self.best_metric

            if (cm > bm) if self.increase else (cm < bm):
                self.best_weights = self.model.get_weights()
                self.best_metric = cm

            if (bm - (bm * self.min_factor) > cm) if self.increase else (bm + (bm * self.min_factor) < cm):
                self.model.set_weights(self.best_weights)
                if self.verbose > 0:
                    print(
                        f"\nValidation metric '{self.monitor}' worsened by more than {self.min_factor}. "
                        f"Rolling back to the previous epoch's weights.")


def Pigeon(count_timeframes=3,
           length_input=100,
           count_features=6,
           count_out=10,
           size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    dens_outputs = []
    for i in range(count_timeframes):
        dens = Dense(size_tail, activation='relu')(input_layer[:, i, :, :])
        dens_outputs.append(dens)

    concatenated = Concatenate(axis=1)(dens_outputs)
    flatten = Flatten()(concatenated)

    dense = Dense(size_tail, activation='relu')(flatten)
    output_layer = Dense(count_out, activation='softmax')(dense)

    optimizer = SGD(learning_rate=1e-2)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])
    return model


def Wyvern(count_timeframes=5,
           length_input=100,
           count_features=6,
           count_out=10,
           size_wing=64,
           size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    conv_outputs = []
    for i in range(count_timeframes):
        iter_input = (input_layer[:, i, :, :])
        # iter_conv = Conv1D(size_wing, 8, activation='relu')(iter_input)
        iter_con = Conv1D(size_wing, 3, activation='relu')(iter_input)
        iter_con = MaxPooling1D(pool_size=2)(iter_con)
        iter_con = Conv1D(size_wing * 2, 3, activation='relu')(iter_con)
        iter_con = MaxPooling1D(pool_size=2)(iter_con)
        iter_con = Conv1D(size_wing * 3, 3, activation='relu')(iter_con)
        iter_con = MaxPooling1D(pool_size=2)(iter_con)
        conv_outputs.append(iter_con)

    concatenated = Concatenate(axis=-1)(conv_outputs)
    flatten = Flatten()(concatenated)

    dense = Dense(size_tail, activation='relu')(flatten)
    drop = Dropout(0.5)(dense)
    output_layer = Dense(count_out, activation='softmax')(drop)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])

    return model


def Chimera(count_timeframes=3,
            length_input=100,
            count_features=6,
            count_out=10,
            size_wing=64,
            size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    conv = None
    for i in range(count_timeframes - 1, -1, -1):
        iter_conv = Conv1D(size_wing, 3, activation='relu', padding='same')(input_layer[:, i, :, :])
        iter_conv = Conv1D(size_wing * 2, 3, activation='relu', padding='same')(iter_conv)
        iter_conv = Conv1D(size_wing * 4, 3, activation='relu', padding='same')(iter_conv)
        if conv is not None:
            conv = Attention()([conv, iter_conv])
        else:
            conv = iter_conv

    flatten = Flatten()(conv)
    dense = Dense(size_tail * 2, activation='relu')(flatten)
    drop = Dropout(0.5)(dense)
    dense = Dense(size_tail, activation='relu')(drop)
    drop = Dropout(0.5)(dense)
    output_layer = Dense(count_out, activation='softmax')(drop)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])
    return model


def Manticore(count_timeframes=3,
              length_input=100,
              count_features=6,
              count_out=10,
              size_wing=64,
              size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    rnn = None
    for i in range(count_timeframes - 1, -1, -1):
        iter_rnn = LSTM(units=size_wing, return_sequences=False)(input_layer[:, i, :, :])
        if rnn is None:
            rnn = iter_rnn
        else:
            rnn = Attention()([rnn, iter_rnn])

    flatten = Flatten()(rnn)
    dense = Dense(size_tail, activation='relu')(flatten)
    drop = Dropout(0.5)(dense)
    output_layer = Dense(count_out, activation='softmax')(drop)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])
    return model


def Eagle(count_timeframes=5,
          length_input=100,
          count_features=6,
          count_out=10,
          size_eye=15,
          size_wing=64,
          size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    outputs = []
    small_outputs = []
    for i in range(count_timeframes):
        iter_in = (input_layer[:, i, :, :])

        iter_con = iter_in
        for c in range(1, 4):
            iter_con = Conv1D(size_wing * c, 3, activation='relu')(iter_con)
            iter_con = MaxPooling1D(pool_size=2)(iter_con)
        iter_con = LSTM(units=size_wing, return_sequences=False)(iter_con)
        outputs.append(iter_con)

        # flatten = Flatten()(iter_in[:, -size_eye:, :])
        # iter_pnn = Dense(size_wing, activation='relu')(flatten)
        iter_pnn = LSTM(units=size_wing, return_sequences=False)(iter_in[:, -size_eye:, :])
        small_outputs.append(iter_pnn)

    concat_cnn = Concatenate(axis=-1)(outputs)
    concat_pnn = Concatenate(axis=-1)(small_outputs)

    concatenated = Concatenate(axis=-1)([concat_cnn, concat_pnn])
    dense = Dense(size_tail, activation='relu')(concatenated)
    output_layer = Dense(count_out, activation='softmax')(dense)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])

    return model


def Hawk(count_timeframes=5,
         length_input=100,
         count_features=6,
         count_out=10,
         size_eye=15,
         size_wing=64,
         size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    outputs = []
    small_outputs = []
    for i in range(count_timeframes):
        iter_in = (input_layer[:, i, :, :])
        iter_con = Conv1D(size_wing, 3, activation='relu')(iter_in)
        iter_con = LSTM(units=size_wing, return_sequences=False)(iter_con)
        outputs.append(iter_con)

        flatten = Flatten()(iter_in[:, -size_eye:, :])
        iter_pnn = Dense(size_wing, activation='relu')(flatten)
        small_outputs.append(iter_pnn)

    concatenated = Concatenate(axis=-1)(outputs)
    dense1 = Dense(size_tail * 2, activation='relu')(concatenated)

    concatenated = Concatenate(axis=-1)(small_outputs)
    dense2 = Dense(size_tail, activation='relu')(concatenated)

    concatenated = Concatenate(axis=-1)([dense1, dense2])
    dense = Dense(size_tail * 4, activation='relu')(concatenated)
    drop = Dropout(0.5)(dense)

    output_layers = []
    for _ in range(int(count_out // 2)):
        output_layer = Dense(2, activation='softmax')(drop)
        output_layers.append(output_layer)
    out_concat = Concatenate()(output_layers)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=out_concat)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])

    return model


def Kite(count_timeframes=5,
         length_input=100,
         count_features=6,
         count_out=10,
         size_wing=64,
         size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    outputs = []

    for i in range(count_timeframes):
        iter_con = (input_layer[:, i, :, :])
        iter_con = Conv1D(size_wing, 3, activation='relu')(iter_con)
        iter_con = MaxPooling1D(pool_size=2)(iter_con)
        iter_con = Conv1D(size_wing * 2, 3, activation='relu')(iter_con)
        iter_con = MaxPooling1D(pool_size=2)(iter_con)
        iter_con = Conv1D(size_wing * 3, 3, activation='relu')(iter_con)
        iter_con = MaxPooling1D(pool_size=2)(iter_con)
        iter_con = Conv1D(size_wing * 4, 3, activation='relu')(iter_con)
        iter_rnn = LSTM(units=size_wing, return_sequences=False)(iter_con)
        outputs.append(iter_rnn)

    concatenated = Concatenate(axis=-1)(outputs)
    dense = Dense(size_tail, activation='relu')(concatenated)
    drop = Dropout(0.5)(dense)
    output_layer = Dense(count_out, activation='softmax')(drop)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])

    return model


def Peacock(count_timeframes=5,
            length_input=100,
            count_features=6,
            count_out=10,
            size_eye=6,
            size_wing=64,
            size_tail=512):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))
    outputs = []

    for i in range(count_timeframes):
        iter_con = (input_layer[:, i, :, :])
        iter_con = (iter_con[:, -size_eye:, :])
        iter_rnn = LSTM(units=size_wing, return_sequences=False)(iter_con)
        outputs.append(iter_rnn)

    if len(outputs) > 1:
        concatenated = Concatenate(axis=-1)(outputs)
    else:
        concatenated = outputs[0]
    dense = Dense(size_tail, activation='relu')(concatenated)
    drop = Dropout(0.5)(dense)
    output_layer = Dense(count_out, activation='softmax')(drop)

    optimizer = Adam(learning_rate=1e-3)
    precision = Precision()

    model = Model(inputs=input_layer, outputs=output_layer)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision])

    return model


def Lera(count_timeframes=5,
         length_input=100,
         count_features=6,
         count_out=1,
         size_wing=64,
         size_tail=256):
    input_layer = Input(shape=(count_timeframes, length_input, count_features))

    outputs = []
    for i in range(count_timeframes):
        iter_con = (input_layer[:, i, :, :])
        # iter_dense = Dense(size_wing, input_dim=count_features, activation='relu')(iter_con)
        # iter_dense = Dense(size_wing * 2, activation='relu')(iter_dense)
        # outputs.append(iter_dense)
        iter_rnn = LSTM(units=size_wing, return_sequences=False)(iter_con)
        iter_dense = Dense(size_wing)(iter_rnn)
        outputs.append(iter_dense)

    if len(outputs) > 1:
        concatenated = Concatenate(axis=-1)(outputs)
    else:
        concatenated = outputs[0]
    output_layer = Dense(count_out, activation='linear')(concatenated)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

    return model


def load(path_model):
    return tf.keras.models.load_model(path_model)


def predict(model,
            input_data):
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    yhat = model.predict(input_data, verbose=0)
    tf.keras.backend.clear_session()
    return yhat


def fit(model,
        input_data,
        output_data,
        epochs=1,
        batch_size=None,
        path_save=None):
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    model.fit(input_data,
              output_data,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1)
    if path_save:
        model.save(path_save)


def draw(model,
         path='model.png',
         open_picture=False):
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
    if open_picture:
        image = Image.open(path)
        image.show()
