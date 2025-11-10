#!/bin/bash

# extract all vocal points of interest in the test-data folder, search between 
# 200 and 1000 Hz and extract if the potential vocalisations are louder than 
# 20 to 30 dB above the median volume of the file of interest. Write the 
# extracted fragments to the file signal.wav and the annotations in a csv 
# file called extracted.csv

data_dir="${1:-data}"

python bioacoustics/wav_processing/condensation/condensate.py --input="$data_dir/original_recordings/mefou" \
    --output-csv="$data_dir/condensed_wav_files/timestamps.csv" \
    --output-signal="$data_dir/condensed_wav_files/test_out.wav" \
    --frequencies="[(200, 1000)]" \
    --volume="(20, 30)"
