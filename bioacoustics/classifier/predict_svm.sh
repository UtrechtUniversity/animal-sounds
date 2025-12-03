data_dir="${1:-data}"
feature_dir="$data_dir/features/"
trained_model="data/models/svm/svm_model.sav
output_dir="$data_dir/predictions/

python bioacoustics/classifier/predict.py --model=svm --feature_dir=$feature_dir --trained_model_path=$trained_model --output_dir=$output_dir

