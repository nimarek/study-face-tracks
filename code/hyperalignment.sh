#!/bin/bash

source /home/data/software/spack-experimental_20230524/ipsy-env/activate

echo "calling python script for sub-$1 roi-$2"
python3 /home/data/study_gaze_tracks/scratch/study_face_tracks/func_alignment_lss.py $1 $2
