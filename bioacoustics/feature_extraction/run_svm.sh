input_dir="../../data/processed_wav_files/"
output_dir="../../output/features/"

echo "Processing $input_dir"

python3 extract_features_svm.py --input_dir $input_dir --output_dir "${output_dir}chimpanze_24000.csv" --frame_length 24000 --hop_length 12000 --cores 8 --filter 100 2000 5 --label1 chimpanze --label2 test
