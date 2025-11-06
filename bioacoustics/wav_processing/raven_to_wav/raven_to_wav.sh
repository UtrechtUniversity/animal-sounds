#!/bin/bash

source .venv/bin/activate
python bioacoustics/wav_processing/raven_to_wav/raven_to_wav.py \
		--annotations_file test_data/annotation_txt_files/test_chimp_annotations.csv \
		--species chimpanzee \
		--wavpath test_data/original_recordings/mefou/ \
		--outputdir test_data/processed_wav_files/ \
		--recID 2C \
		--min_sig_len 0.2 \
		--bg_padding_len 0.05 \
		--startindex 0

