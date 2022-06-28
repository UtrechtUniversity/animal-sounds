#!/bin/bash

# extract all vocal points of interest in the folder /Users/casperkaandorp/Desktop/input_wav_set,
# search between 200 and 1000 Hz and extract if the potential vocalisations are louder than
# 20 to 30 dB above the median volume of the file of interest. Write the extracted fragments to 
# the file signal.wav and the annotations in a csv file called extracted.csv
python extract_vocal_points_of_interest.py --input="/Users/casperkaandorp/Desktop/input_wav_set" --output-csv="extracted.csv" --output-signal="signal.wav" --frequencies="[(200, 1000)]" --volume="(20, 30)"

python extract_vocal_points_of_interest.py --input="./test_data/20191220_190302.WAV" --output-csv="extracted_II.csv" --output-signal="signal.wav" --frequencies="[(200, 1000)]" --volume="(20, 30)"