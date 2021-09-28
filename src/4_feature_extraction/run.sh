#!/bin/bash
# 
#echo "chimps"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/chimpanze/annotations/ --output_dir ~/Desktop/chimps_sanc_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimpanze --label2 sanctuary


# backgrounds
#echo "chimps background"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/chimpanze/background/ --output_dir ~/Desktop/bg_sanc_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 sanctuary

#echo "guenon background"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/guenon/background/ --output_dir ~/Desktop/guenon_bg_sanc_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 sanctuary

#echo "mandrille background"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/mandrille/background/ --output_dir ~/Desktop/mandrille_bg_sanc_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 sanctuary

#echo "chimps background"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/red_cap/background/ --output_dir ~/Desktop/redcap_bg_sanc_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 sanctuary

#echo "jungle features"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/chimpanze/jungle/background/ --output_dir ~/Desktop/jungle_bg_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 jungle

#echo "jungle chimp features"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/version6/chimpanze/jungle/chimpanze/ --output_dir ~/Desktop/jungle_chimp_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimp --label2 jungle

echo "features synth chimp"
python3 main.py --input_dir ~/Repositories/animalsounds/data/synthetic/chimps/ --output_dir ~/Desktop/synth_chimp_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimp --label2 synth

echo "features synth bg"
python3 main.py --input_dir ~/Repositories/animalsounds/data/synthetic/background/ --output_dir ~/Desktop/synth_bg_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 synth


#echo "features test data"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/sanaga_test/chimps/ --output_dir ~/Desktop/test_chimp_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 chimp --label2 test

#echo "features test data bg"
#python3 main.py --input_dir ~/Repositories/animalsounds/data/sanaga_test/background/ --output_dir ~/Desktop/test_bg_24000.csv --frame_length 24000 --hop_length 12000 --cores 10 --filter 100 2000 5 --label1 background --label2 test
