#!/bin/bash

DATADIR='/Volumes/science.data.uu.nl/research-zwerts/data/sanaga_test/chimps/'
OUTPUTDIR='/Volumes/science.data.uu.nl/research-zwerts/data/sanaga_test_chunks/chimps/'
RECORDERS='A1 A3 A4 A5 A21 A22 A26 A38'
echo $DATADIR
for RECORDER in $RECORDERS
do
  echo $DATADIR
  echo $OUTPUTDIR
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/0*.wav' --output_dir $OUTPUTDIR$RECORDER'/0/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/1*.wav' --output_dir $OUTPUTDIR$RECORDER'/1/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/2*.wav' --output_dir $OUTPUTDIR$RECORDER'/2/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/3*.wav' --output_dir $OUTPUTDIR$RECORDER'/3/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/4*.wav' --output_dir $OUTPUTDIR$RECORDER'/4/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/5*.wav' --output_dir $OUTPUTDIR$RECORDER'/5/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/6*.wav' --output_dir $OUTPUTDIR$RECORDER'/6/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/7*.wav' --output_dir $OUTPUTDIR$RECORDER'/7/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/8*.wav' --output_dir $OUTPUTDIR$RECORDER'/8/'
  python3 make_chunks.py --input_dir $DATADIR$RECORDER'/9*.wav' --output_dir $OUTPUTDIR$RECORDER'/9/'
done