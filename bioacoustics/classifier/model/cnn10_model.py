from model.acoustic_model import AcousticModel

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import load_model

class CNN10_model(AcousticModel):

    #num_epochs = 10 #72 500
    #num_batch_size = 32
    #num_channels = 1
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
                Number of epochs
            args[3]: int
                Number of filters in the first Conv layer
            args[4]: int
                Number of filters in the second Conv layer
            args[5]: int
                Dropout percentage
            args[6]: int
                Number of hidden units

        """
        super(CNN10_model, self).__init__()

        self.predicts = None
        if len(args) > 0:
            self.num_rows = args[0]
            self.num_columns = args[1]
            self.num_channels = args[2]
            self.num_epochs = args[3]
            self.batch_size = args[4]
            self.channel_first = args[5]
            self._make_cnn_model()
            self._compile()

    def _make_cnn_model(self):
        """Make a CNN model"""
        if self.channel_first:
            keras.backend.set_image_data_format('channels_first')
            input_shape = (self.num_channels, self.num_rows, self.num_columns)
            data_format = 'channels_first'
        else:
            input_shape = (self.num_rows, self.num_columns, self.num_channels)
            data_format = 'channels_last'

        self.acoustic_model = Sequential()
        self.acoustic_model.add(
            Conv2D(filters=64, kernel_size=3, input_shape=input_shape, data_format=data_format))
        self.acoustic_model.add(BatchNormalization())
        self.acoustic_model.add(Activation('relu'))
        self._cnnBlock(64, data_format)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(0.2))
        self._cnnBlock(128, data_format)
        self._cnnBlock(128, data_format)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(0.2))
        self._cnnBlock(256, data_format)
        self._cnnBlock(256, data_format)
        self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(Dropout(0.2))
        # self._cnnBlock(512, data_format)
        # self._cnnBlock(512, data_format)
        # self.acoustic_model.add(AveragePooling2D(pool_size=2))
        self.acoustic_model.add(GlobalAveragePooling2D())
        self.acoustic_model.add(Dropout(0.5))
        #self.acoustic_model.add(Dense(512, activation='relu'))
        self.acoustic_model.add(Dense(256, activation='relu'))
        self.acoustic_model.add(Dense(self.num_labels, activation='softmax'))



    def _compile(self):
        # Compile the model
        self.acoustic_model.compile(loss='categorical_crossentropy', metrics=[Recall()], optimizer='adam')  # 'accuracy'

        # Display model architecture summary
        self.acoustic_model.summary()

    def _train(self, X_train,y_train,X_test,y_test, file_path):
        """Train a CNN model
            Parameters
            ----------
            file_path: str
                    file path to save the trained model
        """

        # m = X_train.max()
        # X_train = X_train / m
        # X_test = X_test / m

        checkpointer = ModelCheckpoint(filepath=file_path+'_weights.best.cnn.hdf5', #'saved_models/weights.best.basic_cnn.hdf5'
                                       verbose=1, save_best_only=True)
        start = datetime.now()

        print("self.acoustic_model", self.acoustic_model)

        weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}
        history = self.acoustic_model.fit(X_train,
                                y_train,
                                batch_size=self.batch_size,
                                epochs=self.num_epochs,
                                validation_data=(X_test, y_test),
                                shuffle=True,
                                class_weight=weights,
                                callbacks=[checkpointer],
                                verbose=1)
        self.plot_measures(history, file_path,'_CNN10')

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

    def _cnnBlock(self, in_channels, data_format):
        self.acoustic_model.add(
            Conv2D(filters=in_channels, kernel_size=3, data_format=data_format))
        self.acoustic_model.add(BatchNormalization())
        self.acoustic_model.add(Activation('relu'))



