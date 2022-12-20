import argparse
import json
import os
import sys
from pathlib import Path
from random import sample, shuffle

import numpy as np
import pandas as pd
import soundfile as sf

# this hacky bit allows us to include the Extractor
# class from the extractor module in the condensation
# folder
script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
sys.path.append("../condensation")
from extractor import Extractor


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description="Bioacoustics Synth Dataset - step 2, creating an overview"
    )

    # File path to the data.
    parser.add_argument(
        "--input_dir", type=str, help="File path to the dataset of .wav files"
    )

    parser.add_argument("--output", type=str, default=None, help="output file")

    parser.add_argument(
        "--frames", type=int, default=2500, help="number of required output frames"
    )

    parser.add_argument(
        "--frame_length",
        type=int,
        default=4800,
        help="frame length in seconds multiplied by sample rate",
    )

    return parser


if __name__ == "__main__":
    # collect

    # get arguments
    parser = parse_arguments()
    args = vars(parser.parse_args())

    input_files = list(Path(args["input_dir"]).glob("**/*.wav"))
    # shuffle
    shuffle(input_files)

    no_frames = 0
    voc_bucket = []
    # iterate over files
    for f in input_files:
        try:
            voc = Extractor(f)
            # count number of frames
            counted_frames = len(voc.signal) / args["frame_length"]
            # store all
            voc_bucket.append({"path": str(f), "frames": counted_frames})

            no_frames += counted_frames
            print(str(f), no_frames)
        except:
            print("error")
        if no_frames > args["frames"]:
            break

    # ensure folder structure
    out = Path(args["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w+") as out:
        json.dump(voc_bucket, out)
