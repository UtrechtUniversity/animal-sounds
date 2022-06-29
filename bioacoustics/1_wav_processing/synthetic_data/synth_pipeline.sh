#!/bin/bash

echo "collect background"
python 1_collect_background.py \
    --input_dir './test_data/recordings/' \
    --annotation_dir './test_data' \
    --output_dir './test_data/results/background'
echo "create background overview"
python 2_create_overview.py \
    --input_dir './test_data/results/background' \
    --output './test_data/results/overviews/background.json'
echo "create vocalizations overview"
python 2_create_overview.py \
    --input_dir './test_data/vocalizations' \
    --output './test_data/results/overviews/vocalizations.json'
echo "generate synthetic data"
python 3_create_synth_sample.py \
    --primate_json './test_data/results/overviews/vocalizations.json' \
    --background_json './test_data/results/overviews/vocalizations.json' \
    --output './test_data/results/synth_data/'

