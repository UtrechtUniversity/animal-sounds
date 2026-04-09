#!/bin/bash

DATADIR='data/train/'
OUTPUTDIR='output/features/train'
echo $OUTPUTDIR

DATASET='processed_wav_files/'
echo "Processing $DATASET"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'vocalizations/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_chimpanze.pkl' --label 'chimpanze' --config_file "config/testdata.yml"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'background/*/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_background.pkl' --label 'background' --config_file "config/testdata.yml"


DATASET='synth_data/'
echo "Processing $DATASET"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'vocalizations/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_chimpanze.pkl' --label 'chimpanze' --config_file "config/testdata.yml"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'background/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_background.pkl' --label 'background' --config_file "config/testdata.yml"


DATADIR='data/test/processed_wav_files/'
OUTPUTDIR='output/features/test'
echo $OUTPUTDIR


DATASET='14a/'
echo "Processing $DATASET"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'vocalizations/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_chimpanze.pkl' --label 'chimpanze' --config_file "config/testdata.yml"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'background/*/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_background.pkl' --label 'background' --config_file "config/testdata.yml"


DATASET='13b/'
echo "Processing $DATASET"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'vocalizations/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_chimpanze.pkl' --label 'chimpanze' --config_file "config/testdata.yml"
python3 bioacoustics/feature_extraction/extract_features_dl.py --input_dir $DATADIR$DATASET'background/*.wav' --output_dir $OUTPUTDIR'/'$DATASET'_background.pkl' --label 'background' --config_file "config/testdata.yml"
