#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for sub-$1"
python3 /home/data/study_gaze_tracks/scratch/local_code/func_alignment_lss.py $1
