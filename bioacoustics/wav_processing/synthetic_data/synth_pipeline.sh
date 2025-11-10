#!/bin/bash

data_dir="${1:-data}"

echo "collect background"
python bioacoustics/wav_processing/synthetic_data/1_collect_background.py \
    --input_dir "$data_dir/original_recordings/jungle" \
    --annotation_dir "$data_dir/annotation_txt_files/background_jungle" \
    --output_dir "$data_dir/synthetic_intermediate/background"
echo "create background overview"
python bioacoustics/wav_processing/synthetic_data/2_create_overview.py \
    --input_dir "$data_dir/synthetic_intermediate/background" \
    --output "$data_dir/synthetic_intermediate/overviews/background.json"
echo "create vocalizations overview"
python bioacoustics/wav_processing/synthetic_data/2_create_overview.py \
    --input_dir "$data_dir/processed_wav_files/vocalizations" \
    --output "$data_dir/synthetic_intermediate/overviews/vocalizations.json"
echo "generate synthetic data"
python bioacoustics/wav_processing/synthetic_data/3_create_synth_sample.py \
    --primate_json "$data_dir/synthetic_intermediate/overviews/vocalizations.json" \
    --background_json "$data_dir/synthetic_intermediate/overviews/vocalizations.json" \
    --output "$data_dir/synth_data/"

