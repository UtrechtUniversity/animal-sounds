#!/bin/bash

# extract all vocal points of interest in the test-data folder, search between 
# 200 and 1000 Hz and extract if the potential vocalisations are louder than 
# 20 to 30 dB above the median volume of the file of interest. Write the 
# extracted fragments to the file signal.wav and the annotations in a csv 
# file called extracted.csv

python condensate.py --input="./test_data" \
    --output-csv="timestamps.csv" \
    --output-signal="out.wav" \
    --frequencies="[(200, 1000)]" \
    --volume="(20, 30)"