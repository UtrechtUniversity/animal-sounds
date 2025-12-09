data_dir="${1:-data}"
feature_dir="$data_dir/features/"
trained_model="data/models/svm/svm_model.sav"
prediction_set="sample_13b"
output_dir="$data_dir/predictions/"

python bioacoustics/classifier/predict.py --model=svm --feature_dir=$feature_dir --trained_model_path=$trained_model --output_dir=$output_dir --prediction_set=$prediction_set

