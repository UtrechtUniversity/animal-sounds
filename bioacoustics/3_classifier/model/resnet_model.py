from model.acoustic_model import AcousticModel

from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# from model.bioacoustics_generator import BioacousticsGenerator


class RESNET_model(AcousticModel):
    num_labels = 2

    def __init__(self, *args):
        """Class init
            Parameters
            ----------
            args[0]: int
                Number of epochs
        """
        super(RESNET_model, self).__init__()

        self.predicts = None
        if len(args) > 0:
            self.num_epochs = args[0]
            self.batch_size = args[1]
            self._make_resnet_model()

    def _make_resnet_model(self):
        """Make a RESNET model
           https://github.com/Kenneth-ca/holbertonschool-machine_learning/blob/master/supervised_learning/
           0x09-transfer_learning/0-transfer.py
        """

        keras.backend.set_image_data_format('channels_first')

        conv_base = ResNet50(weights='imagenet',
                            include_top=False,
                            input_shape=(3, 64, 64))
        #print(conv_base.summary())

        for layer in conv_base.layers[:143]:
            layer.trainable = False

        for layer in conv_base.layers[143:]:
            layer.trainable = True

        self.acoustic_model = Sequential()
        self.acoustic_model.add(conv_base)
        self.acoustic_model.add(GlobalAveragePooling2D())
        self.acoustic_model.add(Dropout(0.7))
        self.acoustic_model.add(Dense(self.num_labels, activation='softmax'))


        # Compile the model
        #self.acoustic_model.compile(loss='categorical_crossentropy', metrics=[Recall()],
                                    #optimizer=keras.optimizers.RMSprop(learning_rate=2e-5))
        self.acoustic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall()])

        # Display model architecture summary
        #self.acoustic_model.summary()

    def _train(self, X_train,y_train,X_test,y_test,file_path):
        """Train a Resnet model
            Parameters
            ----------
            file_path: str
                    file path to save the trained model
        """
        # m = X_train.max()
        # X_train = X_train / m
        # X_test = X_test / m

        checkpointer = ModelCheckpoint(filepath=file_path+'_weights.best.resnet.hdf5',
                                       verbose=1, save_best_only=True)

        # training_batch_generator = BioacousticsGenerator(X_train, y_train, self.batch_size)
        # validation_batch_generator = BioacousticsGenerator(X_test, y_test, self.batch_size)

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

        self.plot_measures(history, file_path)
        # self.acoustic_model.fit_generator(generator=training_batch_generator,
        #                                   steps_per_epoch=int(X_train.shape[0] // self.batch_size),
        #                                   epochs=self.num_epochs,
        #                                   verbose=1,
        #                                   validation_data=validation_batch_generator,
        #                                   validation_steps=int(X_test.shape[0] // self.batch_size),
        #                                   callbacks = [checkpointer])

        # self.acoustic_model.fit(training_batch_generator.generate_arrays_from_file(),
        #                                   steps_per_epoch=int(X_train.shape[0] // self.batch_size),
        #                                   epochs=self.num_epochs,
        #                                   verbose=1,
        #                                   validation_data=validation_batch_generator.generate_arrays_from_file(),
        #                                   validation_steps=int(X_test.shape[0] // self.batch_size),
        #                                   callbacks=[checkpointer])

        duration = datetime.now() - start
        print("Training completed in time: ", duration)




