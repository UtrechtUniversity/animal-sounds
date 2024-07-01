#!/bin/bash

DATADIR='/Volumes/science.data.uu.nl/research-zwerts/data/sanaga_test_chunks/'
RECORDERS='A1 A3 A4 A5 A21 A22 A26 A38'

OUTPUTDIR='../../output/features/'
echo $DATADIR
for RECORDER in $RECORDERS
do
  echo $DATADIR
  echo $OUTPUTDIR
  python3 extract_features_dl.py --input_dir $DATADIR'chimps/'$RECORDER'/*/*.wav' --output_dir $OUTPUTDIR$RECORDER'/'$RECORDER'_chimpanze.pkl' --label 'chimpanze' --window_length 750  --hop_length 376 --n_mel 64  --new_img_size 64 64
  python3 extract_features_dl.py --input_dir $DATADIR'background/'$RECORDER'/*/*.wav' --output_dir $OUTPUTDIR$RECORDER'/'$RECORDER'_background.pkl' --label 'background' --window_length 750  --hop_length 376 --n_mel 64  --new_img_size 64 64
done


