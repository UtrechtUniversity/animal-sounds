import argparse
import json
import yaml
from pathlib import Path
from random import shuffle

from bioacoustics.wav_processing.condensation.extractor import Extractor


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description="Bioacoustics Synth Dataset - step 2, creating an overview"
    )

    # File path to the data.
    parser.add_argument(
        "--config_file", type=str, help="File path to the config file"
    )

    parser.add_argument(
        "--class", type=str, default=None, help="Class for which to create synthetic data"
    )

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

    with open(args["config_file"], 'r') as f:
        config = yaml.safe_load(f)

    input_dir_temp_background = config["synth_pipeline_vocalizations"]["input_dir_temp_background"]
    input_dir_vocalizations = config["synth_pipeline_vocalizations"]["input_dir_vocalizations"]
    input_dir_background = config["synth_pipeline_vocalizations"]["input_dir_background"]

    if args["class"] == "vocalizations":
        directories = [input_dir_temp_background, input_dir_vocalizations]
    else:
        directories = [input_dir_temp_background, input_dir_background]

    for ix, directory in enumerate(directories):
        input_files = list(Path(directory).glob("**/*.wav", case_sensitive=False))
    
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
            except Exception:
                print("error")
            if no_frames > args["frames"]:
                break

        # ensure folder structure
        if ix == 0:
            out = Path(config["synth_pipeline_vocalizations"]["overview_temp_background"])
        elif args["class"] == "vocalizations":
            out = Path(config["synth_pipeline_vocalizations"]["overview_vocalizations"])
        else:
            out = Path(config["synth_pipeline_vocalizations"]["overview_background"])
        
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w+") as out:
            json.dump(voc_bucket, out)
