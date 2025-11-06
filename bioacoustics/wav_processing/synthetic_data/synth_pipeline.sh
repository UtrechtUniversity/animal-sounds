#!/bin/bash

echo "collect background"
python bioacoustics/wav_processing/synthetic_data/1_collect_background.py \
    --input_dir 'test_data/original_recordings/jungle' \
    --annotation_dir 'test_data/annotation_txt_files/background_jungle' \
    --output_dir 'test_data/synthetic_intermediate/background'
echo "create background overview"
python bioacoustics/wav_processing/synthetic_data/2_create_overview.py \
    --input_dir 'test_data/synthetic_intermediate/background' \
    --output 'test_data/synthetic_intermediate/overviews/background.json'
echo "create vocalizations overview"
python bioacoustics/wav_processing/synthetic_data/2_create_overview.py \
    --input_dir 'test_data/processed_wav_files/vocalizations' \
    --output 'test_data/synthetic_intermediate/overviews/vocalizations.json'
echo "generate synthetic data"
python bioacoustics/wav_processing/synthetic_data/3_create_synth_sample.py \
    --primate_json 'test_data/synthetic_intermediate/overviews/vocalizations.json' \
    --background_json 'test_data/synthetic_intermediate/overviews/vocalizations.json' \
    --output 'test_data/synth_data/'

