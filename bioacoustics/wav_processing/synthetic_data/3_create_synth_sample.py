import argparse
import json
import numpy as np

from pathlib import Path
from random import choice
from pydub import AudioSegment, effects


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description=(
            "Bioacoustics Synth Dataset, step 3: ",
            "creating the synthetic data",
        )
    )
    # File path to the data.
    parser.add_argument(
        "--primate_json",
        type=str,
        help="File path to the overview of vocalization .wav files",
    )
    parser.add_argument(
        "--background_json",
        type=str,
        help="File path to the overview of background .wav files",
    )
    parser.add_argument(
        "--output_folder", type=str, default=None, help="Output directory"
    )
    parser.add_argument(
        "--frame_length",
        type=int,
        default=4800,
        help="frame length in seconds multiplied by sample rate",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=48000, help="sample frequency"
    )
    return parser


def find_suitable_noise_file(voc, bg_set):
    suitable = [x for x in bg_set if x["frames"] >= voc["frames"]]
    return choice(suitable)


if __name__ == "__main__":
    # get arguments
    parser = parse_arguments()
    args = vars(parser.parse_args())

    # create outfolder
    if args["output_folder"] is not None:
        Path(args["output_folder"]).mkdir(parents=True, exist_ok=True)

    frame_length = args["frame_length"]

    # read data
    primate_frames = []
    with open(args["primate_json"], "r") as f:
        primate_frames = json.load(f)

    background_frames = []
    with open(args["background_json"], "r") as f:
        background_frames = json.load(f)

    # sort to get some statistics
    primate_frames = sorted(primate_frames, key=lambda x: x["frames"])
    background_frames = sorted(background_frames, key=lambda x: x["frames"])

    min_primate_frames = primate_frames[0]["frames"]
    max_bg_frames = background_frames[-1]["frames"]
    avg_bg_frames = int(
        sum([x["frames"] for x in background_frames]) / len(background_frames)
    )
    avg_bg_duration = avg_bg_frames * args["frame_length"]

    # read files, files that are too long are chopped up
    primate_set = []
    for i, item in enumerate(primate_frames):

        print("chopping", i + 1)

        path = item["path"]
        frames = item["frames"]
        item["voc"] = AudioSegment.from_file(
            path, format="wav", frame_rate=args["sample_rate"]
        )
        item["postfix"] = ""

        if frames > max_bg_frames:
            no_chunks = int(np.ceil(frames / avg_bg_frames))
            chunk_size = frames / no_chunks

            raw_data = np.frombuffer(item["voc"].raw_data, dtype=np.int16)

            # divide signal in chunks
            signals = np.array_split(raw_data, no_chunks)
            # get signal rate
            sr = args["sample_rate"]
            # if the chunks are bigger than the minimal frame length
            for postfix, sig in enumerate(signals):
                new_frames = len(sig) / frame_length
                if new_frames >= min_primate_frames:

                    new_voc = AudioSegment(
                        data=sig.tobytes(), sample_width=2, frame_rate=sr, channels=1
                    )

                    new_item = {
                        "path": path,
                        "frames": new_frames,
                        "voc": new_voc,
                        "postfix": f"_{postfix}",
                    }
                    primate_set.append(new_item)
        else:
            primate_set.append(item)

    # now we have a primate set for which we can pick background files and produce
    # synthetic samples
    for i, p in enumerate(primate_set):
        print("superpos", i + 1)
        match = find_suitable_noise_file(p, background_frames)
        match["voc"] = AudioSegment.from_file(
            match["path"], format="wav", frame_rate=args["sample_rate"]
        )

        primate_norm = effects.normalize(p["voc"])
        bg_norm = effects.normalize(match["voc"])

        # superposition
        for amp in [0, 3.3, 6.6, 10]:

            softer = primate_norm.apply_gain(-1 * amp)

            superpos = softer.overlay(bg_norm)
            filename = Path(args["output_folder"]) / (
                Path(p["path"]).stem + p["postfix"] + f"_{ amp * 10 }" + ".wav"
            )
            # save
            superpos.export(filename, format="wav")
