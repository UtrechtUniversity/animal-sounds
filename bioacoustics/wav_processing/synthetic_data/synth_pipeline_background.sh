#!/bin/bash

data_dir="${1:-data}"

echo "collect background"
#python bioacoustics/wav_processing/synthetic_data/1_collect_background.py \
#    --input_dir "$data_dir/original_recordings/jungle" \
#    --annotation_dir "$data_dir/annotation_txt_files/background_jungle" \
#    --output_dir "$data_dir/synthetic_intermediate/background"
echo "create overviews"
python bioacoustics/wav_processing/synthetic_data/2_create_overview.py \
    --config_file "config/testdata.yml" \
    --class "background"
echo "generate synthetic data"
python bioacoustics/wav_processing/synthetic_data/3_create_synth_sample.py \
    --primate_json "$data_dir/synthetic_intermediate/overviews/background_test.json" \
    --background_json "$data_dir/synthetic_intermediate/overviews/background_orig_test.json" \
    --output "$data_dir/synth_data/background/"

