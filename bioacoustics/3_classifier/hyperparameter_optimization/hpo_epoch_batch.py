# Use scikit-learn to grid search the dropout rate
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import pandas as pd
import sys
sys.path.append('.')

from model.cnn2_model import CNN2_model

from data_prepration_dl import prepare_data_dl

import os
import argparse

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.constraints import MaxNorm

def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description="Bioacoustics"
    )

    # File path to the data.
    parser.add_argument(
        "--feature_dir",
        type=str,
        help="File path to the dataset of features"
    )

    parser.add_argument(
        "--normVal_dir",
        type=str,
        help="File path to the mean and std values of trained data to normalize test dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='cnn10',
        help="machine learning model "
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output dir'
    )

    return parser

# Function to create model, required for KerasClassifier


def create_model():
    # """Make a CNN model"""

    s = CNN2_model(64, 64, 1, True)

    c_model = s.make_model()

    print(type(c_model))

    return c_model

    # num_channels=1
    # num_rows=64
    # num_columns=64
    # weight_constraint = 1
    # dropout_rate =0.2
    # num_labels=2
    #
    # keras.backend.set_image_data_format('channels_first')
    # input_shape = (num_channels, num_rows, num_columns)
    # data_format = 'channels_first'
    #
    #
    # acoustic_model = Sequential()
    # acoustic_model.add(
    #     Conv2D(filters=64, kernel_size=3, input_shape=input_shape,
    #            activation='relu', data_format=data_format, padding='same',
    #            kernel_regularizer=regularizers.l2(l=0.01), kernel_initializer='uniform', #init_mode,
    #            kernel_constraint=MaxNorm(weight_constraint)))
    #
    # acoustic_model.add(Conv2D(filters=64, kernel_size=3, activation='relu',
    #                                kernel_regularizer=regularizers.l2(l=0.01), kernel_initializer='uniform', #init_mode,
    #                                kernel_constraint=MaxNorm(weight_constraint)))
    #
    # acoustic_model.add(Dropout(dropout_rate))
    # acoustic_model.add(MaxPooling2D(pool_size=2))
    # acoustic_model.add(Flatten())
    # acoustic_model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l=0.01),
    #                               kernel_initializer='uniform', #init_mode
    #                         kernel_constraint=MaxNorm(weight_constraint)))
    # acoustic_model.add(Dropout(dropout_rate))
    # acoustic_model.add(Dense(num_labels, activation='softmax', kernel_initializer='uniform')) #init_mode
    #
    # # Compile model
    # acoustic_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall()])
    # print(type(acoustic_model))
    # return acoustic_model

parser = parse_arguments()
args = parser.parse_args()

if not os.path.exists(os.path.dirname(args.output_dir)):
    os.makedirs(os.path.dirname(args.output_dir))

# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)

X_train, y_train, X_test, y_test = prepare_data_dl(args.feature_dir, num_channels=1,
                                                   normval_dir=args.normVal_dir, fraction=0.4)

# create model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = KerasClassifier(model=create_model, verbose=1) #callbacks=[callback],

# define the grid search parameters
batch_size = [64] #8,32,
epochs = [1, 10] #, 50
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

d = {'mean':means, 'stdev':stds, 'param':params}
pd.DataFrame(data=d).to_csv(args.output_dir + '_params.csv', index=False)

d = {'best_score':[grid_result.best_score_], 'best_params':[grid_result.best_params_]}
pd.DataFrame(data=d).to_csv(args.output_dir + '_best_params.csv', index=False)