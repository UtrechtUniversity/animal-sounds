# 1_wav_processing

## [raven_to_wav](./raven_to_wav)
Raven to wav is used to filter annotated parts from the original `.wav` files based on annotations in table format (e.g. `.txt` or `.csv`).

## [Condensation](./condensation)
In condensation we use an energy change based method to filter out low energy parts from the original dataset to make manual annotation/labeling more efficient.

## [Synthetic data](./synthetic_data)
In Synthetic data we embed Chimpanze vocalizations in jungle sounds that are labeled as background to create more and more diverse data.

## [Chunk wav](./chunk_wav)
Chunk_wav component splits `.wav` files into smaller chunks. This is a step in the preparation process for deep 
learning models. By default, desired length of wav files is 0.5 seconds. There is an overlap of 0.25 seconds between each of two 
following chunks.

Find usage steps in the respective folders.
