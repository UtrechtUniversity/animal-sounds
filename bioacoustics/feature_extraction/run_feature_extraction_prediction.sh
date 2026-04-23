#!/bin/bash
# Usage: ./bioacoustics/feature_extraction/run_feature_extraction_svm.sh 

config_file="${1:-config/testdata.yml}"

for job_name in chimpanze_mefou background_mefou; do
	echo "Processing $job_name"
	python bioacoustics/feature_extraction/extract_features_svm.py --config_file "$config_file" --job_name "$job_name"
done
