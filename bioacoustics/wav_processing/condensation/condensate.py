import argparse
import pandas as pd

from pathlib import Path
from extractor import Extractor


def parse_arguments():
    """Parses command line parameters"""
    parser = argparse.ArgumentParser(
        description="Command line parameters for the extraction script"
    )
    # Path to the folder that contains the text files
    parser.add_argument(
        "--input",
        type=str,
        help="Input path <str>, directory, file or, if comma separated, multiple files",
    )
    # Path to output csv file
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path of csv file that contains extraction data (timestamps)",
    )
    # Path to output signal file
    parser.add_argument(
        "--output-signal",
        type=str,
        help="Path of wav file that contains all extracted fragments",
    )
    # Frequencies of interest
    parser.add_argument(
        "--frequencies",
        type=str,
        default="[(200, 5000)]",
        help=(
            (
                "String containing a list of tuples containing frequency bands "
                "of interest. Default is [(200, 5000)] denoting a single band between "
                "200Hz en 5000Hz."
            )
        ),
    )
    # Sound volume of interest
    parser.add_argument(
        "--volume",
        type=str,
        default="(0,40)",
        help=(
            (
                "String to specify a volume band with respect to a base value, "
                "which is typically the median volume, in which vocalizations of "
                "interest are expected. A tuple (x,y) denotes a volume range "
                "between xdB and ydB with respect to the base volume. If a single "
                "value N is given, a range between 0dB and NdB will be assumed."
            )
        ),
    )
    return parser


if __name__ == "__main__":

    parser = parse_arguments()
    args = vars(parser.parse_args())

    # collect input from CLI
    if "," in args["input"]:
        parsed_input = [Path(f.strip()) for f in args["input"].split(",")]
    elif Path(args["input"]).is_file():
        parsed_input = [Path(args["input"])]
    elif Path(args["input"]).is_dir():
        parsed_input = Path(args["input"]).glob("*.WAV")
    else:
        raise RuntimeError(f"input is not a file or folder")

    # collect frequencies of interest from CLI
    freqs = eval(args["frequencies"])

    # collect relative volumes of interest from CLI
    volume = eval(args["volume"])
    if type(volume) in [int, float]:
        min_vol = 0
        max_vol = volume
    elif type(volume) is tuple:
        min_vol = volume[0]
        max_vol = volume[1]
    else:
        raise RuntimeError(
            f"volume must be a single value of tuple containing 2 values"
        )

    # list used to collect the csv data
    csv_data = []
    # signal_out is used to store all potentials vocalization
    signal_out = None

    counter = 0
    result_counter = 1
    csv_stopwatch = 0.0

    for filepath in sorted(parsed_input):

        try:

            print(f"{counter} - {filepath.name}")

            # create extractor object for current wav file
            extr = Extractor(filepath)

            # detect potential vocalizations
            pois = extr.detect_vocalizations(
                freqs=freqs,
                min_threshold_db=min_vol,
                max_threshold_db=max_vol,
                threshold_pattern=0.1,
                ignore_voc=0.09,
                padding=0.35,
            )

            # extract th potential vocalisations from the wav file
            padding = 0.1
            extracted = extr.extract_intervals(pois, padding=padding)

            # add extracted signal to signal_out
            signal_out = extracted if signal_out is None else signal_out + extracted

            # process peaks (potential vocalisations) and add to csv
            rows = []
            for p in pois:
                # duration of vocalisation
                voc_duration = p[1] - p[0]
                # start
                res_start = round(csv_stopwatch, 3)
                res_end = round(csv_stopwatch + voc_duration, 3)

                csv_stopwatch = csv_stopwatch + voc_duration + padding

                # this is for the csv file
                rows.append(
                    (
                        str(filepath),
                        filepath.name,
                        p[0],
                        p[1],
                        f'{args["output_csv"]}',
                        res_start,
                        res_end,
                    )
                )

            csv_data += rows

            # increase counter
            counter += 1

        except Exception:
            print(f"bad file: {filepath}")

    # write data and signal_out to files
    signal_out.to_wav(Path(args["output_signal"]))
    df = pd.DataFrame(
        csv_data,
        columns=[
            "path",
            "filename",
            "t_start",
            "t_end",
            "result_file",
            "res_start",
            "res_end",
        ],
    )
    df.to_csv(Path(args["output_csv"]), index=False)
