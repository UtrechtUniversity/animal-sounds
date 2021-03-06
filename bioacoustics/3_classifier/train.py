"""Script to apply a model on a set of features_old and make a prediction"""

from data_prepration_dl import prepare_data_dl
from data_prepration_svm import prepare_data_svm
from model.svm_model import SVM_model
from model.cnn_model import CNN_model
from model.cnn10_model import CNN10_model
from model.cnn6_model import CNN6_model
from model.resnet_model import RESNET_model
import os
import argparse
import pandas as pd


# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# import sys
# from data_prepration_large_data import prepare_large_data
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
        default='cnn',
        help="machine learning model "
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output dir'
    )

    ##### params for hyperparameter optimization
    parser.add_argument(
        '-T',
        type=int,
        default=-1,
        help='id for grid-search'
    )
    parser.add_argument(
        '--nrow_input',
        type=int,
        default=64,
        help='first dimension of input'
    )
    parser.add_argument(
        '--ncol_input',
        type=int,
        default=64,
        help='second dimension of input'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='number of epochs'
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        default=3,
        help='number of channels'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size'
    )
    parser.add_argument(
        '--channel_first',
        type=bool,
        default=False,
        help='indicate if the channel is the first dimension'
    )
    return parser


def main():
    # cv_results = False
    parser = parse_arguments()
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_dir)):
        os.makedirs(os.path.dirname(args.output_dir))

    if args.model == 'svm':
        X_train, y_train, X_test, y_test = prepare_data_svm(args.feature_dir,
                                                            args.output_dir)

    else:
        X_train, y_train, X_test, y_test = prepare_data_dl(args.feature_dir, num_channels=args.num_channels,
                                                           normval_dir=args.normVal_dir)

    if args.model == 'resnet':
        # X_train, y_train, X_test, y_test = prepare_large_data(args.feature_dir, args.output_dir)
        # X_train = preprocess_input(X_train)
        # X_test = preprocess_input(X_test)
        s = RESNET_model(args.epochs, args.batch_size)
    elif args.model == 'cnn':
        s = CNN_model(args.nrow_input, args.ncol_input, args.num_channels, args.epochs, args.batch_size,
                      args.channel_first)

    elif args.model == 'cnn10':
        s = CNN10_model(args.nrow_input, args.ncol_input, args.num_channels, args.epochs, args.batch_size,
                        args.channel_first)

    elif args.model == 'cnn6':
        s = CNN6_model(args.nrow_input, args.ncol_input, args.num_channels, args.epochs, args.batch_size,
                       args.channel_first)

    elif args.model == 'svm':
        s = SVM_model()
        # cv_results = True

    print(" X_train.shpe", X_train.shape)
    print(" y_train.shpe", y_train.shape)
    print(" X_test.shpe", X_test.shape)
    print(" y_test.shpe", y_test.shape)

    s.apply_model(X_train, y_train, X_test, y_test, args.output_dir)
    s.save_results(y_test, args.output_dir)  # , cv_results)


# execute main function
if __name__ == "__main__":
    main()
