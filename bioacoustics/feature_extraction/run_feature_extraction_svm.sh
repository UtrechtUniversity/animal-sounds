
data_dir="${1:-data}"

input_dir="$data_dir/processed_wav_files/vocalizations/"
output_dir="$data_dir/features/"

echo "Processing $input_dir"

python bioacoustics/feature_extraction/extract_features_svm.py --input_dir $input_dir --output_dir "${output_dir}chimpanze_24000.csv" --frame_length 24000 --hop_length 12000 --cores 4 --filter 100 2000 5 --label1 chimpanze --label2 test
