"""
Evaluate a trained model on labeled data
"""

from data_preparation_dl import prepare_data_dl
from data_preparation_svm import prepare_data_svm
from model.cnn10_torch import CNN10Model
from model.cnn12_torch import CNN12Model
from model.svm_model import SVM_model
import yaml
import argparse
import logging
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bioacoustics")

    parser.add_argument(
        "--config_file", type=str, help="File path to the config file"
    )
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    init_mode = config["model_setting"]["init_mode"]
    num_channels = config["model_setting"]["num_channels"]
    dropout_rate = config["model_setting"]["dropout_rate"]
    weight_constraint = config["model_setting"]["weight_constraint"]

    frames_per_chunk = config["model_setting"]["frames_per_chunk"]
    batch_size = config["model_setting"]["batch_size"]

    model = config["evaluate"]["model"]
    output_dir = config["evaluate"]["output_dir"]
    feature_dir = config["evaluate"]["feature_dir"]
    model_dir = config["evaluate"]["model_dir"]
    aco_model = None

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    if model == "svm":
        logging.info("SVM model")
        _, _, x_test, y_test = prepare_data_svm(
            features_path=feature_dir, output_dir=output_dir,
            trained_model_path=model_dir
        )
        aco_model = SVM_model(output_dir=output_dir, model_dir=model_dir)
        accuracy, preds = aco_model.evaluate(x_test, y_test)
        aco_model.save_evaluation(preds, y_true=y_test, metrics=accuracy,split="eval")

    else:
        logging.info("DL model")
        _, _, _, _,  x_test, y_test = prepare_data_dl(
            features_dir=feature_dir, mode="evaluate",
        )
        if model == "cnn10":
            aco_model = CNN10Model(num_channels=num_channels, output_dir=output_dir, model_dir=model_dir,
                                   init_mode=init_mode, dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        elif model == "cnn12":
            aco_model = CNN12Model(num_channels=num_channels, output_dir=output_dir, model_dir=model_dir,
                                   init_mode=init_mode, dropout_rate=dropout_rate, weight_constraint=weight_constraint)

        chunk_results, file_results = aco_model.evaluate(x_test, y_test, frames_per_chunk=frames_per_chunk,
                                                         batch_size=batch_size)
        aco_model.save_evaluation(
            probs=chunk_results['probs'],
            predicts=chunk_results['preds'],
            y_true=chunk_results['y_true'],
            metrics=chunk_results['metrics'],
            split="eval_chunk")
        aco_model.save_evaluation(
            probs=file_results['probs'],
            predicts=file_results['preds'],
            y_true=file_results['y_true'],
            metrics=file_results['metrics'],
            split="eval_file")


if __name__ == "__main__":
    main()
