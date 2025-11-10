#!/bin/bash
data_dir="${1:-data}"
source .venv/bin/activate
python bioacoustics/wav_processing/raven_to_wav/raven_to_wav.py \
		--annotations_file "$data_dir/annotation_txt_files/test_chimp_annotations.csv" \
		--species chimpanzee \
		--wavpath "$data_dir/original_recordings/mefou/" \
		--outputdir "$data_dir/processed_wav_files/" \
		--recID 2C \
		--min_sig_len 0.2 \
		--bg_padding_len 0.05 \
		--startindex 0

