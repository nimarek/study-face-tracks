#!/bin/bash

logs_dir=/home/data/study_gaze_tracks/code/logs_roi-hyperalignment/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

ROIS=(
    ctx-lh-bankssts ctx-rh-bankssts
    ctx-lh-caudalanteriorcingulate ctx-rh-caudalanteriorcingulate
    ctx-lh-caudalmiddlefrontal ctx-rh-caudalmiddlefrontal
    ctx-lh-cuneus ctx-rh-cuneus
    ctx-lh-entorhinal ctx-rh-entorhinal
    ctx-lh-fusiform ctx-rh-fusiform
    ctx-lh-inferiorparietal ctx-rh-inferiorparietal
    ctx-lh-inferiortemporal ctx-rh-inferiortemporal
    ctx-lh-insula ctx-rh-insula
    ctx-lh-isthmuscingulate ctx-rh-isthmuscingulate
    ctx-lh-lateraloccipital ctx-rh-lateraloccipital
    ctx-lh-lateralorbitofrontal ctx-rh-lateralorbitofrontal
    ctx-lh-lingual ctx-rh-lingual
    ctx-lh-medialorbitofrontal ctx-rh-medialorbitofrontal
    ctx-lh-middletemporal ctx-rh-middletemporal
    ctx-lh-parahippocampal ctx-rh-parahippocampal
    ctx-lh-paracentral ctx-rh-paracentral
    ctx-lh-parsopercularis ctx-rh-parsopercularis
    ctx-lh-parsorbitalis ctx-rh-parsorbitalis
    ctx-lh-parstriangularis ctx-rh-parstriangularis
    ctx-lh-pericalcarine ctx-rh-pericalcarine
    ctx-lh-postcentral ctx-rh-postcentral
    ctx-lh-posteriorcingulate ctx-rh-posteriorcingulate
    ctx-lh-precentral ctx-rh-precentral
)

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 4
request_memory = 8G

# Execution
initial_dir    = /home/data/study_gaze_tracks/scratch/study_face_tracks/
executable     = hyperalignment.sh
\n"

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
    for roi in "${ROIS[@]}"; do
        printf "arguments = ${sub} ${roi}\n"
        printf "log       = ${logs_dir}/sub-${sub}_roi-${roi}\$(Cluster).\$(Process).log\n"
        printf "output    = ${logs_dir}/sub-${sub}_roi-${roi}\$(Cluster).\$(Process).out\n"
        printf "error     = ${logs_dir}/sub-${sub}_roi-${roi}\$(Cluster).\$(Process).err\n"
        printf "Queue\n\n"
    done
done