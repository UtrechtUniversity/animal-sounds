#!/bin/bash
# version7/chimpanze
echo "chimps sanct"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version7/chimpanze/sanctuary/ --output_dir ~/Desktop/chimps_sanc_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimpanze --label2 sanctuary

echo "chimp synth jungle"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version7/chimpanze/synthetic/jungle_bg/ --output_dir ~/Desktop/chimps_synt_jungle_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimpanze --label2 synth


# version7/background
echo "chimps background"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version7/background/sanctuary_chimp/ --output_dir ~/Desktop/bg_sanc_chimp_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 sanctuary

echo "background sanctuary other"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version7/background/sanctuary_other/ --output_dir ~/Desktop/bg_sanc_other_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 sanctuary

echo "synthetic background"
python3 main.py --input_dir ~/Repositories/animalsounds/data/version7/background/synthetic/ --output_dir ~/Desktop/bg_synth_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 synth

#echo "features test data"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/sanaga_test/chimps/ --output_dir ~/Desktop/test_chimp_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimp --label2 test

#echo "features test data bg"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/sanaga_test/background/ --output_dir ~/Desktop/test_bg_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 test
