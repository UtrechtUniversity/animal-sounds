"""Script to apply a model on a set of features_old and make a prediction"""

from data_prepration_dl import prepare_data
from model.svm_model import SVM_model
from model.cnn_model import CNN_model
from model.cnn10_model import CNN10_model
from model.cnn6_model import CNN6_model
from model.resnet_model import RESNET_model
#import statistics
import os
import argparse
# import pandas as pd
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
#import sys

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
        "--trained_model_path",
        type=str,
        default='',
        help="file path of a pre-trained model to apply on a given dataset"
    )

    parser.add_argument(
        '--without_label',
        type=bool,
        default=False,
        help='indicate if dataset is labeled'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output dir'
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        default=3,
        help='number of channels'
    )

    return parser


def main():
    dl_model = True
    parser = parse_arguments()
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_dir)):
        os.makedirs(os.path.dirname(args.output_dir))

    X_train, y_train, X_test, y_test = prepare_data(args.feature_dir, without_label=args.without_label,
                                                    trained_model_path=args.trained_model_path,
                                                    num_channels=args.num_channels,
                                                    normval_dir= args.normVal_dir)

    if args.model =='resnet':
        s = RESNET_model()
    elif args.model =='cnn':
        s = CNN_model()
    elif args.model == 'cnn10':
        s = CNN10_model()
    elif args.model == 'cnn6':
        s = CNN6_model()

    if args.model=='svm':
        s = SVM_model()
        dl_model = False

    #s.evaluate_model( X_test, y_test, args.trained_model_path)
    s.predict_model(X_test, args.trained_model_path, dl_model)

    if args.without_label: # apply model on un-labeled dataset
        s.save_results(y_test, args.output_dir, predicts_only=True)
    else:
        s.save_results(y_test, args.output_dir)

# execute main function
if __name__ == "__main__":
    main()

