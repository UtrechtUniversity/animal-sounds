"""Script to train and test a model on a set of features"""

from data_preparation_dl import prepare_data_dl
from data_preparation_svm import prepare_data_svm
from model.svm_model import SVM_model
from model.cnn10_torch import CNN10Model
from model.cnn12_torch import CNN12Model
import logging
import os
import argparse
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bioacoustics")

    parser.add_argument(
        "--config_file", type=str, help="File path to the config file"
    )

    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    acoustic_model = None

    logging.basicConfig(level=logging.INFO)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    model = config["model_training"]["model"]
    output_dir = config["model_training"]["output_dir"]
    augmentation = config["model_training"]["augmentation"]
    set_threshold = config["model_training"]["set_threshold"]
    num_channels = config["model_setting"]["num_channels"]
    epoch = config["model_setting"]["epochs"]
    batch_size = config["model_setting"]["batch_size"]
    dropout_rate = config["model_setting"]["dropout_rate"]
    weight_constraint = config["model_setting"]["weight_constraint"]
    learning_rate = config["model_setting"]["learning_rate"]
    init_mode = config["model_setting"]["init_mode"]
    weight_decay = config["model_setting"]["weight_decay"]
    samples_per_epoch = config["model_setting"]["samples_per_epoch"]
    frames_per_chunk = config["model_setting"]["frames_per_chunk"]
    feature_dir = config["feature_extraction"]["feature_dir"]

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    if model == "svm":
        logging.info("SVM model")
        x_train, y_train, x_test, y_test = prepare_data_svm(
            feature_dir, output_dir
        )

    else:
        logging.info("DL model")
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_data_dl(
            feature_dir, mode="train"
        )
        logging.info("preprocessing is Done!")

    if model == "cnn10":
        acoustic_model = CNN10Model(num_channels=num_channels, output_dir=output_dir, init_mode=init_mode,
                                    dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    elif model == "cnn12":
        acoustic_model = CNN12Model(num_channels=num_channels, output_dir=output_dir, init_mode=init_mode,
                                    dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    elif model == "svm":
        acoustic_model = SVM_model(output_dir=output_dir)

    if model != "svm":
        acoustic_model.fit(
            x_train, y_train, x_val, y_val,
            epoch=epoch, batch_size=batch_size,learning_rate=learning_rate, weight_decay=weight_decay,
            frames_per_chunk=frames_per_chunk, samples_per_epoch=samples_per_epoch, augmentation=augmentation,
            set_threshold=set_threshold
        )

        chunk_results, file_results = acoustic_model.evaluate_model(x_test, y_test, batch_size=batch_size,
                                                                    frames_per_chunk=frames_per_chunk,
                                                                    tune_threshold=False)
        acoustic_model.save_evaluation(probs=chunk_results['probs'], y_true=chunk_results["y_true"],
                                       metrics=chunk_results["metrics"], predicts=chunk_results["preds"],
                                       split="test_chunks")
        acoustic_model.save_evaluation(probs=file_results['probs'], y_true=file_results["y_true"],
                                       metrics=file_results["metrics"], predicts=file_results["preds"],
                                       split="test_files")

    else:
        acoustic_model.fit(x_train, y_train)
        accuracy, preds = acoustic_model.evaluate_model(x_test, y_test)
        acoustic_model.save_evaluation(preds, y_true=y_test, metrics=accuracy, split="test")


if __name__ == "__main__":
    main()
