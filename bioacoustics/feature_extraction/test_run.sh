input_dir="./test_input/"
output_dir="./test_output/"

echo "Processing $input_dir"

python3 extract_features_svm.py --input_dir $input_dir --output_dir "${output_dir}test_24000.csv" --frame_length 24000 --hop_length 12000 --cores 1 --filter 100 2000 5 --label1 chimpanze --label2 test
