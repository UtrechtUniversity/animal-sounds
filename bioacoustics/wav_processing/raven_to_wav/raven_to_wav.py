import pandas as pd
import librosa
import soundfile as sf
import os
import argparse


def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(description="Process Raven files")
    # File path to the data.
    parser.add_argument(
        "annotations_file", type=str, help="File path to the annotation dataset"
    )
    # Annotation class
    parser.add_argument(
        "species", type=str, help="Only process annotation of this class"
    )
    # Raw files location
    parser.add_argument("wavpath", type=str, help="Location of raw wav files")
    parser.add_argument("outputdir", type=str, help="Output directory")
    parser.add_argument("recID", type=str, help="Recorder Identifier")
    parser.add_argument("min_sig_len", type=float, help="Minimal Signal Length (s)")
    parser.add_argument(
        "bg_padding_len", type=float, help="Standard Background Padding Length (s)"
    )

    parser.add_argument(
        "startindex", type=int, default=0, help="Start index for numbering output files"
    )
    parser.add_argument(
        "-c",
        "--createframes",
        action="store_true",
        dest="createframes",
        help="Cut annotations into multiple frames",
    )

    return parser


class ProcessRaven:
    def __init__(self, file, wav_path):
        self.filelist = sorted(os.listdir(wav_path))
        self.df = None
        self.df_2 = None
        self.df_pad = None
        self.duration = None
        self.file = file
        self.file_lengths = None
        self.wav_path = wav_path

    def read_raven(self):
        """Read Raven annotations .txt file.

        Reads Raven annotations file as pandas dataframe and
        removes uppercase characters in column names

        Returns
        -------
        DataFrame
            Raven annotations in dataframe format
        """
        if self.file[-3:] == "csv":
            self.df = pd.read_csv(self.file)
        else:
            self.df = pd.read_table(self.file)
        self.df.columns = self.df.columns.str.lower()
        print(self.df.columns)

    def compute_file_lengths(self, wav_path):
        """Compute file lenghts of .WAV files.

        Lists .wav files from a specific folder,
        gets the length of the files in seconds and
        combines them in a dataframe

        Parameters
        ----------
        wav_path : str
            Location of .wav files

        Returns
        -------
        DataFrame
            Filenames and file lenghts in seconds
        """
        len_file_list = []
        for f in self.filelist:
            len_file = librosa.get_duration(filename=wav_path + f)
            len_file_list.append(len_file)

        df = pd.DataFrame(columns=["file", "filelength"])
        df["file"], df["filelength"] = self.filelist, len_file_list
        self.file_lengths = df

    @staticmethod
    def split_multifile_annotations(df, filelist, wav_path):
        """Split annotations that are composed of multiple .wav files.

        Annotations that are composed of multiple .wav files are split into
        multiple annotations that will replace the original annotation in
        the returned datafrome. The original rows are dropped and new rows
        appended at the end of the dataframe in the following order:
        - segment from file where the annotation starts
        - segment from file where the annotation ends
        - full files between start and end (when annotation spans more than 2 .wav files)

        Parameters
        ----------
        df : DataFrame
            dataframe containing annotated segments

        filelist : list of str
            list containing all .wav filenames of annotated dataset.

        wav_path : str
            location of the .wav files


        Returns
        -------
        DataFrame
            Cleaned dataframe
        """
        row_i = list(df[(df["file"] != df["end_file"])].index)

        for i in row_i:
            df_temp = pd.DataFrame(
                [
                    [
                        df.loc[i, "file"],
                        df.loc[i, "file"],
                        df.loc[i, "start_time"],
                        df.loc[i, "filelength"],
                        df.loc[i, "filelength"],
                    ],
                    [
                        df.loc[i, "end_file"],
                        df.loc[i, "end_file"],
                        0.0,
                        df.loc[i, "end_time"] - df.loc[i, "filelength"],
                        df.loc[i, "filelength"],
                    ],
                ],
                columns=["file", "end_file", "start_time", "end_time", "filelength"],
            )

            df = df.append(df_temp, ignore_index=True)

            if (
                filelist.index(df.loc[i, "end_file"])
                - filelist.index(df.loc[i, "file"])
            ) > 1:

                for j in range(
                    filelist.index(df.loc[i, "file"]) + 1,
                    filelist.index(df.loc[i, "end_file"]),
                ):
                    print(j)
                    len_file = librosa.get_duration(filename=wav_path + filelist[j])
                    df_temp = pd.DataFrame(
                        [[filelist[j], filelist[j], 0.0, len_file, len_file]],
                        columns=[
                            "file",
                            "end_file",
                            "start_time",
                            "end_time",
                            "filelength",
                        ],
                    )
                df = df.append(df_temp, ignore_index=True)
        return df.drop(row_i)

    @staticmethod
    def select_columns_rows(df, species, min_sig_len):
        """Select rows of interest.

        This function selects rows from the input DataFrame
        that contain annotations from the species of interest
        and are longer than the defined minimum signal length (s).

        Parameters
        ----------
        df : DataFrame
            dataframe containing annotated segments

        species : str
            Species (or class) of interest

        min_sig_len : float
            Minimal threshold signal length. Annotations shorter than
            this threshold will be excluded.
        """
        return df.loc[
            (df["species"] == species)
            & ((df["end time (s)"] - df["begin time (s)"]) > min_sig_len),
            ["file", "end_file", "start_time", "end_time", "type", "filelength"],
        ]

    def process_raven(self, species, min_sig_len):
        """Parse annotations dataframe and select relevant parts.

        This function performs processing steps of the annotations
        table. Columns are renamed, paths are stripped from filenames,
        annotations spanning multiple .wav files are split into
        separate annotations and annotation durations are calculated.

        Parameters
        ----------
        species : str
            Species (or class) of interest

        min_sig_len : float
            Minimal threshold signal length. Annotations shorter than
            this threshold will be excluded.
        """
        df = self.df.rename(columns={"begin path": "file", "end path": "end_file"})

        # select only relevant colums
        if "class" in df.columns:
            df = df.rename(columns={"class": "species"})

        # Change timestamps to timestamps relative to file
        df["start_time"] = df["file offset (s)"]
        df["end_time"] = (
            df["file offset (s)"] + df["end time (s)"] - df["begin time (s)"]
        )

        # Cut first part of string
        df["file"] = df["file"].str.replace(".*\\\\", "", regex=True)
        df["end_file"] = df["end_file"].str.replace(".*\\\\", "", regex=True)

        df = pd.merge(df, self.file_lengths, on="file")

        df = self.split_multifile_annotations(df, self.filelist, self.wav_path)

        df_temp = self.select_columns_rows(df, species, min_sig_len)

        df_temp = df_temp.reset_index()
        self.df_2 = df_temp.drop(columns=["index"])
        self.duration = self.df_2["end_time"] - self.df_2["start_time"]

    def padding(self, df, wav_path, min_bg_padding, min_length):
        """Background padding.

        Pads vocalizations on two sides with unannotated segments of
        specified length. When this is not entirely possible because the audio
        file starts or ends within this unannotated segment, the segment
        on the other end will be extended with the remaining padding length.

        Parameters
        ----------
        df : DataFrame
            dataframe containing annotated segments

        wav_path : str
            location of the .wav files

        min_bg_padding : float
            background padding length in seconds.
            This length will be added in front as well as
            behind the annotated segment

        min_length : float
            annotation will be extended with background padding
            to a minimum specified by this length. If the
            annotated vocalization combined with min_bg_padding
            will not reach this minimum length the background segments
            will be extended to reach this minimum
        """
        self.df_pad = pd.DataFrame({"file": [], "start": [], "duration": []})
        for index, row in df.iterrows():
            file = wav_path + row.begin_path
            duration = max(
                min_length,
                min(row.end_time - row.start_time + 2 * min_bg_padding, 60.0),
            )
            if duration < row.filelength - 2 * min_bg_padding:
                bg_padding = max(
                    min_bg_padding, (min_length - (row.end_time - row.start_time)) / 2
                )
                start = max(row.start_time - bg_padding, 0.0) - max(
                    0.0, (row.end_time + bg_padding) - row.filelength
                )
            else:
                start = 0.0
                duration = 60.0
            print(duration)
            self.df_pad = self.df_pad.append(
                {"file": file, "start": start, "duration": duration}, ignore_index=True
            )
        self.df_pad["type"] = list(df["type"])
        self.df_pad["filename"] = list(df["file"])


def write_files(
    df, output_path, species, rec_id, createframes, k_start, frame_len, jump_len
):
    """Write annotations to new .wav files

    Read annotated audio signal from the original .wav files and write
    .wav files containing annotated signal only.

    Parameters
    ----------
    df : DataFrame
        dataframe containing annotated segments

    output_path : str
        location where to store the .wav files

        This description can span multiple lines and/or paragraphs as
        well.
    param2 : str, optional
        another description
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    k = k_start
    kk = 0  # second fileindex for output files (when n raven files > 1)
    df2 = pd.DataFrame(
        columns=["id", "recorder_id", "file", "start", "duration", "type", "wav"]
    )

    for index in range(len(df)):
        file = df["file"].iloc[index]
        filename = df["filename"].iloc[index]
        print(index, file[0:-4])
        y, sr = librosa.load(
            file,
            sr=48000,
            offset=df["start"].iloc[index],
            duration=df["duration"].iloc[index],
        )

        if createframes and df["duration"].iloc[index] >= frame_len + jump_len:
            frames = librosa.util.frame(
                y, frame_length=frame_len * sr, hop_length=jump_len * sr, axis=0
            )
            nframes = len(frames)
            # nframes = int(np.floor(df['duration'].iloc[index] - jump_len))
            print(nframes)
            for i in range(nframes):
                outfile1 = (
                    str(k)
                    + "_"
                    + str(index)
                    + "_"
                    + filename[0:-4]
                    + "_"
                    + str(round(df["start"].iloc[index] + i, 6))
                    + ".wav"
                )
                out_file = output_path + outfile1
                sf.write(out_file, frames[i], sr)

                df2.at[kk, "id"] = k
                df2.iloc[kk, 1:6] = [
                    rec_id,
                    filename,
                    str(round(df["start"].iloc[index] + i, 6)),
                    frame_len,
                    df["type"].iloc[index],
                ]
                df2.at[kk, "wav"] = outfile1
                k = k + 1
                kk = kk + 1

        else:
            outfile1 = (
                str(k)
                + "_"
                + rec_id
                + "_"
                + filename[0:-4]
                + "_"
                + str(round(df["start"].iloc[index], 6))
                + ".wav"
            )
            out_file = output_path + outfile1
            sf.write(out_file, y, sr)
            df2.at[kk, "id"] = k
            df2.iloc[kk, 1:6] = [
                rec_id,
                filename,
                str(round(df["start"].iloc[index], 6)),
                df["duration"].iloc[index],
                df["type"].iloc[index],
            ]
            df2.at[kk, "wav"] = outfile1
            k = k + 1
            kk = kk + 1

    if k_start == 0:
        df2.to_csv(output_path + species.lower() + ".csv")
    else:
        df2.to_csv(output_path + species.lower() + ".csv", mode="a", header=False)


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    p1 = ProcessRaven(args.annotations_file, args.wavpath)
    p1.read_raven()
    p1.compute_file_lengths(args.wavpath)
    p1.process_raven(args.species, args.min_sig_len)
    p1.padding(p1.df_2, args.wavpath, args.bg_padding_len, min_length=0.5)
    write_files(
        p1.df_pad,
        args.outputdir,
        args.species,
        args.recID,
        args.createframes,
        args.startindex,
        frame_len=2,
        jump_len=1,
    )


if __name__ == "__main__":
    main()
