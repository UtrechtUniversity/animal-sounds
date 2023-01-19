"""Script of a class with 10 nn blocks"""
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import MaxNorm
from model.acoustic_model import AcousticModel


class CNN10Model(AcousticModel):
    """A class for acoustic model with 10 nn blocks"""
    num_labels = 2

    def __init__(self, *args):
        """Class init
        Parameters
        ----------
        args[0]: int
            Number of rows
        args[1]: int
            Number of columns
        args[2]: int
            Number of channels
        args[3]: bool
            Indicates if the number-of-channels is the first input or not
        """
        super().__init__()  # CNN10Model, self
        self.acoustic_model = None
        self.predicts = None

        if len(args) > 0:
            self.num_rows = args[0]
            self.num_columns = args[1]
            self.num_channels = args[2]
            self.channel_first = args[3]

    def _make_cnn_model(self, init_mode, dropout_rate, weight_constraint):
        """Make a CNN model"""
        if self.channel_first:
            keras.backend.set_image_data_format("channels_first")
            input_shape = (self.num_channels, self.num_rows, self.num_columns)
            data_format = "channels_first"
        else:
            input_shape = (self.num_rows, self.num_columns, self.num_channels)
            data_format = "channels_last"

        self.acoustic_model = Sequential()
        self.acoustic_model.add(
            Conv2D(
                filters=64,
                kernel_size=3,
                input_shape=input_shape,
                data_format=data_format,
                padding="same",
                #kernel_regularizer=regularizers.l2(l=0.1),  # 0.01
                kernel_initializer=init_mode,
                kernel_constraint=MaxNorm(weight_constraint),
            )
        )
        self.acoustic_model.add(BatchNormalization())
        self.acoustic_model.add(Activation("relu"))
        self._cnn_block(64, data_format, init_mode, weight_constraint)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(dropout_rate))  # Dropout(0.2)
        self._cnn_block(128, data_format, init_mode, weight_constraint)
        self._cnn_block(128, data_format, init_mode, weight_constraint)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(dropout_rate))  # Dropout(0.2)
        self._cnn_block(256, data_format, init_mode, weight_constraint)
        self._cnn_block(256, data_format, init_mode, weight_constraint)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(dropout_rate))  # Dropout(0.2)

        self._cnn_block(512, data_format, init_mode, weight_constraint)
        self._cnn_block(512, data_format, init_mode, weight_constraint)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(dropout_rate)) ## add new
        self.acoustic_model.add(GlobalAveragePooling2D())
        self.acoustic_model.add(Dropout(dropout_rate))  # Dropout(0.5)

        self.acoustic_model.add(
            Dense(
                512,
                activation="relu",
                kernel_initializer=init_mode,
                kernel_constraint=MaxNorm(weight_constraint),
            )
        )

        self.acoustic_model.add(Dropout(dropout_rate))  # new added
        # self.acoustic_model.add(Dense(256, activation='relu'))
        self.acoustic_model.add(
            Dense(self.num_labels, activation="softmax",
                  kernel_initializer=init_mode)
        )

    def _train(self,
               x_train,
               y_train,
               x_test,
               y_test,
               file_path, epochs,
               batch_size):
        """Train a CNN model
        Parameters
        ----------
        file_path: str
                file path to save the trained model
        """

        checkpointer = ModelCheckpoint(
            filepath=file_path + "_weights.best.cnn.hdf5",
            verbose=1,
            save_best_only=True,
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        start = datetime.now()

        print("self.acoustic_model", self.acoustic_model)

        weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}
        history = self.acoustic_model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            class_weight=weights,
            callbacks=[checkpointer,callback],
            verbose=1,
        )
        self.plot_measures(history, file_path, "_CNN10")

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

    def _cnn_block(self,
                   in_channels,
                   data_format,
                   init_mode,
                   weight_constraint):
        self.acoustic_model.add(
            Conv2D(
                filters=in_channels,
                kernel_size=3,
                data_format=data_format,
                padding="same",
                #kernel_regularizer=regularizers.l2(l=0.1),  # 0.01
                kernel_initializer=init_mode,
                kernel_constraint=MaxNorm(weight_constraint),
            )
        )
        self.acoustic_model.add(BatchNormalization())
        self.acoustic_model.add(Activation("relu"))
