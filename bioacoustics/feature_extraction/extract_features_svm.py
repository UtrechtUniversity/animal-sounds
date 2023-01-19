import argparse
import asyncio
import multiprocessing as mp
import pandas as pd
import numpy as np
import random
import time

from pathlib import Path
from itertools import product

from acoustic_features.config import Config
from acoustic_features.features import FeatureVector
from acoustic_features.LLD import LLD


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(description="Bioacoustics_features")

    # File path to the data.
    parser.add_argument(
        "--input_dir", type=str, help="File path to the dataset of .wav files"
    )

    parser.add_argument(
        "--sample", type=int, default=0, help="sample n files from folder"
    )

    parser.add_argument("--output_dir", type=str, default=None, help="output dir")

    parser.add_argument("--cores", type=int, default=1, help="number of cores")

    parser.add_argument("--frame_length", type=int, default=1200, help="frame_length")

    parser.add_argument("--hop_length", type=int, default=480, help="hop_length")

    parser.add_argument("--sample_rate", type=int, default=48000, help="sample rate")

    parser.add_argument("--filter", nargs="+", default=[], help="filter")

    parser.add_argument("--label1", type=str, default="-", help="first label")

    parser.add_argument("--label2", type=str, default="-", help="first label")

    return parser


def main(workload):
    path = "config/features/features_01.json"
    config = Config(path)
    config.read()
    features = FeatureVector(config)

    workload, args = workload
    cores = args["cores"]
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
    cores = mp.cpu_count()
    print("Number of processors on your machine: ", cores)

    # get arguments
    parser = parse_arguments()
    args = vars(parser.parse_args())
    args["filter"] = tuple(args["filter"])

    # required cores
    cores = args["cores"]
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
        result = pool.map_async(main, product(workload, [args]))
        result = pd.concat(result.get())
    t2 = time.time()
    print(f"Processed in {t2 - t1} sec")

    # add extra labels
    result.insert(loc=1, column="label_1", value=args["label1"])
    result.insert(loc=2, column="label_2", value=args["label2"])

    if result is not None:
        result = result.sort_values(by=["file_path", "frameId"])
        result.to_csv(args["output_dir"], index=False)
        result.reset_index(inplace=True, drop=True)
