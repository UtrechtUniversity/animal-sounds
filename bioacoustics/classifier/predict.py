"""Script to apply a model on a set of features_old and make a prediction"""
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
    # parse arguments if available
    parser = argparse.ArgumentParser(description="Bioacoustics")

    # File path to the data.
    parser.add_argument(
        "--config_file", type=str, help="File path to the config file"
    )
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    init_mode = config["model_setting"]["init_mode"]
    num_channels = config["model_setting"]["num_channels"]
    dropout_rate = config["model_setting"]["dropout_rate"]
    weight_constraint = config["model_setting"]["weight_constraint"]

    frames_per_chunk = config["model_setting"]["frames_per_chunk"]
    batch_size = config["model_setting"]["batch_size"]

    model = config["predict"]["model"]
    output_dir = config["predict"]["output_dir"]
    feature_dir = config["predict"]["feature_dir"]
    model_dir = config["predict"]["model_dir"]
    aco_model = None

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    if model == "svm":
        logging.info("SVM model")
        _, _, x_pred, _ = prepare_data_svm(
            features_path=feature_dir, output_dir=output_dir,
            trained_model_path=model_dir
        )
        aco_model = SVM_model(output_dir=output_dir, model_dir=model_dir)
        preds = aco_model.predict_model(x_pred)
        aco_model.save_evaluation(preds, split="preds")

    else:
        logging.info("DL model")
        _, _, _, _, x_pred, _ = prepare_data_dl(
            features_dir=feature_dir, mode="predict",
        )

        if model == "cnn10":
            aco_model = CNN10Model(num_channels=num_channels, output_dir=output_dir, model_dir=model_dir,
                           init_mode=init_mode, dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        elif model == "cnn12":
            aco_model = CNN12Model(num_channels=num_channels, output_dir=output_dir, model_dir=model_dir,
                           init_mode=init_mode, dropout_rate=dropout_rate, weight_constraint=weight_constraint)

        results = aco_model.predict_model(x_pred, frames_per_chunk=frames_per_chunk, batch_size=batch_size)
        chunk_results = results["chunk_results"]
        file_results = results["file_results"]

        aco_model.save_evaluation(
            probs=chunk_results['probs'],
            predicts=chunk_results['preds'],
            split="predict_chunk")
        aco_model.save_evaluation(
            probs=file_results['probs'],
            predicts=file_results['preds'],
            split="predict_file")


if __name__ == "__main__":
    main()
