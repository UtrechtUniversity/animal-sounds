import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.append("..")
from condensation.extractor import Extractor


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description="Bioacoustics Synth Dataset, step 1: fragment collection"
    )

    # File path to the data.
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to folder that contains the jungle recordings",
    )

    parser.add_argument(
        "--annotation_dir",
        type=str,
        help=(
            "Path to folder that contains (a) Raven file(s)",
            "with background noise annotations",
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="folder to collect the output (WAV fragments)",
    )

    parser.add_argument(
        "--one_file",
        type=str,
        default=False,
        help="save output in one file (when True) or multiple files (default)",
    )

    return parser


def collect_filename_from_raven_path(path):
    return path.split("\\")[-1]


# COLLECT FRAGMENTS ACCORDING TO CONTENTS OF RAVEN file
if __name__ == "__main__":

    parser = parse_arguments()
    args = vars(parser.parse_args())

    one_file = args["one_file"]

    k = 0

    all_raven_files = list(Path(args["annotation_dir"]).glob("*.txt"))
    out_path = Path(args["output_dir"])
    # create folders if necessary
    out_path.mkdir(parents=True, exist_ok=True)

    result = Extractor()
    result.sr = 48000

    for raven_file in all_raven_files:

        # get all relevant files
        p = Path(args["input_dir"]).glob("**/*.WAV")
        filenames = sorted(list(p))

        # get rid of weird file
        filenames = [f.name for f in filenames if not f.name.startswith(".")]

        # read and clean Raven file
        bg = pd.read_csv(raven_file, delimiter="\t")
        bg["Notes"] = bg["Notes"].apply(lambda x: "" if pd.isnull(x) else x)
        bg["Notes"] = bg["Notes"].str.strip()
        bg["Begin Path"] = bg["Begin Path"].apply(
            lambda x: Path(args["input_dir"], collect_filename_from_raven_path(x))
        )
        bg["End Path"] = bg["End Path"].apply(
            lambda x: Path(args["input_dir"], collect_filename_from_raven_path(x))
        )

        bg = bg[bg["Class"] == "Background"]

        # iterate over all observations
        for index, row in bg.iterrows():
            print(index)

            # minute time
            start_time = row["Begin Time (s)"] % 60.0
            end_time = row["End Time (s)"] % 60.0

            print(row["Begin Path"])

            if row["Begin Path"] != row["End Path"]:

                # collect all relevant files
                index_s = filenames.index(Path(row["Begin Path"]).name)
                index_e = filenames.index(Path(row["End Path"]).name)
                files_involved = filenames[index_s: (index_e + 1)]

                # iterate over all involved files of this record
                for i, name in enumerate(files_involved):
                    sample = Extractor(Path(args["input_dir"]) / name)
                    if i == 0:
                        extract = sample.extract_intervals([(start_time, 60.0)])
                        if one_file:
                            result += extract
                        else:
                            outfile1 = (
                                str(k)
                                + "_"
                                + files_involved[0][0:-4]
                                + "_"
                                + str(round(start_time, 6))
                                + ".wav"
                            )
                            out_file = out_path + outfile1
                            extract.to_wav(out_file)
                    elif name == files_involved[-1]:
                        extract = sample.extract_intervals([(0, end_time)])
                        if one_file:
                            result += extract
                        else:
                            outfile1 = (
                                str(k)
                                + "_"
                                + files_involved[-1][0:-4]
                                + "_"
                                + str(0.0)
                                + ".wav"
                            )
                            out_file = out_path + outfile1
                            extract.to_wav(out_file)

                    else:
                        # whole file
                        if one_file:
                            result += sample
                        else:
                            outfile1 = (
                                str(k)
                                + "_"
                                + files_involved[i][0:-4]
                                + "_"
                                + str(0.0)
                                + ".wav"
                            )
                            out_file = out_path + outfile1
                            sample.to_wav(out_file)
                    k = k + 1
            else:
                print(row["Begin Path"])
                sample = Extractor(Path(row["Begin Path"]))
                extract = sample.extract_intervals([(start_time, end_time)])

                outfile1 = (
                    str(k)
                    + "_"
                    + Path(row["Begin Path"]).name[0:-4]
                    + "_"
                    + str(round(start_time, 6))
                    + ".wav"
                )
                out_file = Path(out_path, outfile1)
                extract.to_wav(out_file)
                k = k + 1
