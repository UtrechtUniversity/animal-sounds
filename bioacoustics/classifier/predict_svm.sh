feature_dir='../../output/features/svm/sanaga/A26/'
trained_model='../../output/models/svm/all/svm_model.sav'
output_dir='../../output/models/svm/all/predictions/A26test/'

python3.8 predict.py --model=svm --feature_dir=$feature_dir --trained_model_path=$trained_model --output_dir=$output_dir

