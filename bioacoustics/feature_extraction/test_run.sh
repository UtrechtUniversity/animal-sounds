config_file="${1:-../../config/testdata.yml}"
job_name="test"

echo "Processing $job_name"

python3 extract_features_svm.py --config_file "$config_file" --job_name "$job_name"
