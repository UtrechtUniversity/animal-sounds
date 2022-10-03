#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate animalsounds

#python3.8 raven_to_wav.py /home/jelle/Repositories/animalsounds/data/raven_annotations/version7/mefou-trainingdata/chimps/20191220_165800.Table.2.selections.chimps.revised_2021_07_15.txt Chimpanzee /home/jelle/Repositories/animalsounds/data/mefou/2/chimpanze/ /home/jelle/Repositories/animalsounds/data/version7/chimps/ 2C 0.2 0.05 0

#python3.8 raven_to_wav.py /home/jelle/Repositories/animalsounds/data/raven_annotations/version7/mefou-trainingdata/chimps/extracted_chimps_set1_set4_Table.1.selections.Revised_2021_07_15.txt Chimpanzee /home/jelle/Repositories/animalsounds/data/mefou/condensed_files/ /home/jelle/Repositories/animalsounds/data/version7/chimps/ 2C 0.2 0.05 112

#python3.8 raven_to_wav.py /home/jelle/Repositories/animalsounds/data/raven_annotations/version7/mefou-trainingdata/background/20191220_165800.Table.1.selections.chimps.background.txt Background /home/jelle/Repositories/animalsounds/data/mefou/2/chimpanze/ /home/jelle/Repositories/animalsounds/data/version7/background/sanctuary/ 2C 0.2 0.05 0

python3.8 raven_to_wav.py /home/jelle/Repositories/animalsounds/data/raven_annotations/version7/sanaga-evaluationdata/20210218_081037.Table.1.selections.SanagaYong.A6.complete_JZ_2.txt Chimpanzee /home/jelle/Repositories/animalsounds/data/mefou/2/chimpanze/ /home/jelle/Repositories/animalsounds/data/version7/chimps/ 2C 0.2 0.05 0
