import argparse
import asyncio
import multiprocessing as mp
import pandas as pd
import numpy as np
import random
import time
import yaml

from pathlib import Path
from itertools import product

from bioacoustics.feature_extraction.acoustic_features.config import Config
from bioacoustics.feature_extraction.acoustic_features.features import FeatureVector
from bioacoustics.feature_extraction.acoustic_features.LLD import LLD


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(description="Bioacoustics_features")

    parser.add_argument(
        "--config_file", type=str, default=None, help="File path to the config file"
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Name of the SVM extraction job defined in the config file",
    )

    # File path to the data.
    parser.add_argument(
        "--input_dir", type=str, default=None, help="File path to the dataset of .wav files"
    )

    parser.add_argument(
        "--sample", type=int, default=0, help="sample n files from folder"
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output CSV path"
    )

    parser.add_argument("--cores", type=int, default=None, help="number of cores")

    parser.add_argument("--frame_length", type=int, default=None, help="frame_length")

    parser.add_argument("--hop_length", type=int, default=None, help="hop_length")

    parser.add_argument("--sample_rate", type=int, default=None, help="sample rate")

    parser.add_argument("--filter", nargs="+", default=None, help="filter")

    parser.add_argument("--label1", type=str, default=None, help="first label")

    parser.add_argument("--label2", type=str, default=None, help="second label")

    return parser


def load_runtime_settings(arguments):
    config = {}
    svm_config = {}
    job_config = {}

    if arguments["config_file"] is not None:
        with open(arguments["config_file"], "r") as file_pointer:
            config = yaml.safe_load(file_pointer) or {}

        svm_config = config.get("feature_extraction_svm", {})
        if arguments["job_name"] is not None:
            job_config = svm_config.get("jobs", {}).get(arguments["job_name"])
            if job_config is None:
                raise KeyError(
                    f"Unknown SVM feature extraction job '{arguments['job_name']}'"
                )

    feature_config = config.get("feature_extraction", {})
    defaults = svm_config.get("defaults", {})

    settings = {
        "input_dir": arguments["input_dir"] or job_config.get("input_dir"),
        "output_dir": arguments["output_dir"] or job_config.get("output_dir"),
        "sample": arguments["sample"],
        "cores": arguments["cores"]
        if arguments["cores"] is not None
        else defaults.get("cores"),
        "frame_length": arguments["frame_length"]
        if arguments["frame_length"] is not None
        else defaults.get("frame_length", 1200),
        "hop_length": arguments["hop_length"]
        if arguments["hop_length"] is not None
        else defaults.get("hop_length", 480),
        "sample_rate": arguments["sample_rate"]
        if arguments["sample_rate"] is not None
        else defaults.get("sample_rate", feature_config.get("sample_rate", 48000)),
        "filter": arguments["filter"] if arguments["filter"] is not None else defaults.get("filter", []),
        "label1": arguments["label1"] or job_config.get("label1") or defaults.get("label1", "-"),
        "label2": arguments["label2"] or job_config.get("label2") or defaults.get("label2", "-"),
    }

    if settings["input_dir"] is None:
        raise ValueError("An input directory must be provided via CLI or config")
    if settings["output_dir"] is None:
        raise ValueError("An output path must be provided via CLI or config")

    settings["filter"] = tuple(settings["filter"])
    return settings


def main(workload):
    path = "bioacoustics/feature_extraction/config/features/features_01.json"
    config = Config(path)
    config.read()
    features = FeatureVector(config)

    workload, cores = workload

    # chop up the workload into chunks
    max_open = int(200 / cores)
    workload = [workload[x : x + max_open] for x in range(0, len(workload), max_open)]
    lld = LLD(
        workload,
        frame_length=args["frame_length"],
        hop_length=args["hop_length"],
        sr=args["sample_rate"],
        bandpass_filter=args["filter"],
        config=config,
        features=features,
    )
    res = asyncio.run(lld.extract())
    return res


# make sure every core gets roughly the same amount of Megabytes to handle
def balance_workload(all_files, cores):
    # order reversed based on file_size
    all_files = sorted(all_files, key=lambda x: x.stat().st_size, reverse=True)
    # make sure we have anough items to form a matrix
    all_files = np.array(all_files + ([None] * (cores - (len(all_files) % cores))))
    # split into a matrix
    workload = all_files.reshape(len(all_files) // cores, cores)
    return workload.T


if __name__ == "__main__":

    # get arguments
    parser = parse_arguments()
    args = load_runtime_settings(vars(parser.parse_args()))

    cores = args["cores"]
    # required cores
    if cores is None:
        cores = max(mp.cpu_count() - 2, 1)

    print("Number of processors on your machine: ", mp.cpu_count()) 
    print(f"Running on {cores} cores.")

    t1 = time.time()
    # get input path
    all_files = list(Path(args["input_dir"]).glob("**/*.[wW][aA][vV]"))
    all_files = [fp for fp in all_files if not fp.name.startswith(".")]

    # sample if necessary
    if args["sample"] > 0:
        all_files = random.sample(all_files, args["sample"])

    # divide the workload
    workload = balance_workload(all_files, cores)
    t2 = time.time()
    print(f"Read {len(all_files)} files in {t2 - t1} sec")

    # start a pool
    t1 = time.time()
    result = None
    with mp.Pool(processes=cores) as pool:
        # do it
        result = pool.map_async(main, product(workload, [cores]))
        result = pd.concat(result.get())
    t2 = time.time()
    print(f"Processed in {t2 - t1} sec")

    # add extra labels
    result.insert(loc=1, column="label_1", value=args["label1"])
    result.insert(loc=2, column="label_2", value=args["label2"])

    if result is not None:
        result = result.sort_values(by=["file_path", "frameId"])
        Path(args["output_dir"]).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args["output_dir"], index=False)
        result.reset_index(inplace=True, drop=True)
