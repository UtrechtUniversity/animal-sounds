#!/bin/bash

python3.8 raven_to_wav.py --annotations_file="table.txt"
		--species=Chimpanzee 
		--wavpath=/data/original_wav_files/ 
		--outputdir=/data/processed_wav_files/ 
		--recID=2C 
		--min_sig_len=0.2 
		--bg_padding_len=0.05 
		--createframes=0

