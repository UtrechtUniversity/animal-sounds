#!/bin/bash
# Usage: ./bioacoustics/feature_extraction/run_feature_extraction_svm.sh 

data_dir="${1:-predict_data}"
output_dir="$data_dir/features/"

# Run for real chimp data

input_dir="$data_dir/processed_wav_files/vocalizations/"

echo "Processing $input_dir"

python bioacoustics/feature_extraction/extract_features_svm.py --input_dir $input_dir --output_dir "${output_dir}chimpanze_mefou_24000.csv" --frame_length 24000 --hop_length 12000 --cores 4 --filter 100 2000 5 --label1 chimpanze --label2 train

# Run for background data
input_dir="$data_dir/processed_wav_files/background/"

echo "Processing $input_dir"

python bioacoustics/feature_extraction/extract_features_svm.py --input_dir $input_dir --output_dir "${output_dir}background_mefou_24000.csv" --frame_length 24000 --hop_length 12000 --cores 4 --filter 100 2000 5 --label1 background --label2 train
