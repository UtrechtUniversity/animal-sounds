""" Module that uses scikit-learn for grid search on the dropout rate """
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import pandas as pd
import sys
import os
import argparse

sys.path.append("..")

from model.cnn10_model import CNN10_model
from data_preparation_dl import prepare_data_dl


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(description="Bioacoustics")

    # File path to the data.
    parser.add_argument(
        "--feature_dir", type=str, help="File path to the dataset of features"
    )

    parser.add_argument(
        "--normVal_dir",
        type=str,
        help="File path to the mean and std values of trained data to normalize test dataset",
    )
    parser.add_argument(
        "--model", type=str, default="cnn10", help="machine learning model "
    )

    parser.add_argument("--output_dir", type=str, default=None, help="output dir")

    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")

    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    return parser


# Function to create model, required for KerasClassifier


def create_model(init_mode="uniform", dropout_rate=0.2, weight_constraint=1):
    # """Make a CNN model"""

    s = CNN10_model(64, 64, 1, True)

    model = s.make_model(
        init_mode=init_mode,
        dropout_rate=dropout_rate,
        weight_constraint=weight_constraint,
        compile_model=False,
    )

    return model


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_dir)):
        os.makedirs(os.path.dirname(args.output_dir))

    # fix random seed for reproducibility
    seed = 7
    tf.random.set_seed(seed)

    X_train, y_train, X_test, y_test = prepare_data_dl(
        args.feature_dir, num_channels=1, normval_dir=args.normVal_dir, fraction=0.4
    )

    # create model
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    model = KerasClassifier(
        model=create_model,
        loss="categorical_crossentropy",
        optimizer="adam",
        callbacks=[callback],
        epochs=args.epoch,
        batch_size=args.batch_size,
        verbose=1,
    )

    # define the grid search parameters
    learn_rate = [0.001, 0.01, 0.1]
    param_grid = dict(optimizer__learning_rate=learn_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    d = {"mean": means, "stdev": stds, "param": params}
    pd.DataFrame(data=d).to_csv(args.output_dir + "_params.csv", index=False)

    d = {"best_score": [grid_result.best_score_], "best_params": [grid_result.best_params_]}
    pd.DataFrame(data=d).to_csv(args.output_dir + "_best_params.csv", index=False)


if __name__ == "__main__":
    main()
