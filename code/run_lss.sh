#!/bin/bash

sub_list=("01" "02" "03" "04" "06" "10" "14" "15" "16" "17" "18" "19" "20")
export script_path="/home/exp-psy/Desktop/study_face_tracks/code/func_alignment_lss.py"

# start jobs
printf "%s\n" "${sub_list[@]}" | parallel -j 3 python $script_path {}
